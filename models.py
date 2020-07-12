#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class MLPNet(nn.Module):
    """
    Vanilla multi-layer perceptron for classification.
    """

    def __init__(self, configs):
        super(MLPNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        self.num_classes = configs["num_classes"]

    def forward(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        h_relu = self.softmax(h_relu)
        return F.log_softmax(h_relu, dim=1)


class FairNet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for fairness.
    """

    def __init__(self, configs):
        super(FairNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_classes = configs["num_classes"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
        self.num_adversaries_layers = len(configs["adversary_layers"])
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                          for i in range(self.num_adversaries_layers)])
        self.sensitive_cls = nn.Linear(self.num_adversaries[-1], 2)

    def forward(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        h_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            h_relu = F.relu(adversary(h_relu))
        cls = F.log_softmax(self.sensitive_cls(h_relu), dim=1)
        return logprobs, cls

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs


class CFairNet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for conditional fairness.
    """
    def __init__(self, configs):
        super(CFairNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_classes = configs["num_classes"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
        self.num_adversaries_layers = len(configs["adversary_layers"])
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])

    def forward(self, inputs, labels):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = labels == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        return logprobs, c_losses

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs
