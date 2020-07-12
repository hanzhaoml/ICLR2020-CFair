#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AdultDataset
from models import MLPNet, FairNet, CFairNet
from utils import conditional_errors
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="adult")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the adversarial classification loss",
                    type=float, default=10.0)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=100)
parser.add_argument("-r", "--lr", type=float, help="Learning rate of optimization", default=1.0)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=512)
parser.add_argument("-m", "--model", help="Which model to run: [mlp|fair|cfair|cfair-eo]", type=str,
                    default="mlp")
parser.add_argument("-y", "--target", help="Name of the target attribute", type=str, default="income")
parser.add_argument("-p", "--private", help="Name of the sensitive/private attribute", type=str, default="sex")
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

logger = get_logger(args.name)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dtype = np.float32

logger.info("UCI Adult data set, target attribute: {}, sensitive attribute: {}".format(args.target, args.private))
# Load UCI Adult dataset.
time_start = time.time()
adult_train = AdultDataset(root_dir='data', phase='train', tar_attr=args.target, priv_attr=args.private)
adult_test = AdultDataset(root_dir='data', phase='test', tar_attr=args.target, priv_attr=args.private)
train_loader = DataLoader(adult_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(adult_test, batch_size=args.batch_size, shuffle=False)
time_end = time.time()
logger.info("Time used to load all the data sets: {} seconds.".format(time_end - time_start))
input_dim = adult_train.xdim
num_classes = 2
num_groups = 2

# Pre-compute the statistics in the training set.
train_target_attrs = np.argmax(adult_train.A, axis=1)
train_target_labels = np.argmax(adult_train.Y, axis=1)
train_idx = train_target_attrs == 0
train_base_0, train_base_1 = np.mean(train_target_labels[train_idx]), np.mean(train_target_labels[~train_idx])
train_y_1 = np.mean(train_target_labels)
# For reweighing purpose.
if args.model == "cfair":
    reweight_target_tensor = torch.tensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1]).to(device)
elif args.model == "cfair-eo":
    reweight_target_tensor = torch.tensor([1.0, 1.0]).to(device)
reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).to(device)
reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).to(device)
reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

logger.info("Average value of A = {}".format(np.mean(train_target_attrs)))
logger.info("A: Male = 0, Female = 1")
# Pre-compute the statistics in the test set.
target_insts = torch.from_numpy(adult_test.X).float().to(device)
target_labels = np.argmax(adult_test.Y, axis=1)
target_attrs = np.argmax(adult_test.A, axis=1)
test_idx = target_attrs == 0
conditional_idx = target_labels == 0
base_0, base_1 = np.mean(target_labels[test_idx]), np.mean(target_labels[~test_idx])
label_marginal = np.mean(target_labels)
logger.info("Value of Base 0: {}, value of Base 1: {}".format(base_0, base_1))
# Compute the conditional probability of Pr(A = 0 | Y = 1)
attr_label = np.mean(np.logical_and(test_idx, ~conditional_idx)) / np.mean(~conditional_idx) 
cls_error, error_0, error_1 = 0.0, 0.0, 0.0
# Pr(Pred = 1 | A = 0, 1)
pred_0, pred_1 = 0.0, 0.0
# Pr(Pred = 1 | A = 0, 1, Y = 0, 1)
cond_00, cond_01, cond_10, cond_11 = 0.0, 0.0, 0.0, 0.0

# Train MLPNet to get baseline results.
configs = {"num_classes": num_classes, "num_groups": num_groups, "num_epochs": args.epoch,
           "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "input_dim": input_dim,
           "hidden_layers": [60], "adversary_layers": [50]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]
lr = configs["lr"]
if args.model == "mlp":
    logger.info("Experiment without debiasing:")
    logger.info("Hyperparameter setting = {}.".format(configs))
    # Train MLPNet without debiasing.
    time_start = time.time()
    net = MLPNet(configs).to(device)
    logger.info("Model architecture: {}".format(net))
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    net.train()
    for t in range(num_epochs):
        running_loss = 0.0
        for xs, ys, attrs in train_loader:
            xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
            optimizer.zero_grad()
            ypreds = net(xs)
            # Compute prediction accuracy on training set.
            loss = F.nll_loss(ypreds, ys)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        logger.info("Iteration {}, loss value = {}".format(t, running_loss))
    time_end = time.time()
    logger.info("Time used for training = {} seconds.".format(time_end - time_start))
    # Test.
    net.eval()
    preds_labels = torch.max(net(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
elif args.model == "fair":
    # Training with FairNet to show the debiased results.
    logger.info("Experiment with FairNet: {} adversarial debiasing:".format(args.model))
    logger.info("Hyperparameter setting = {}.".format(configs))
    time_start = time.time()
    net = FairNet(configs).to(device)
    logger.info("Model architecture: {}".format(net))
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    mu = args.mu
    net.train()
    for t in range(num_epochs):
        running_loss, running_adv_loss = 0.0, 0.0
        for xs, ys, attrs in train_loader:
            xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
            optimizer.zero_grad()
            ypreds, apreds = net(xs)
            # Compute both the prediction loss and the adversarial loss.
            loss = F.nll_loss(ypreds, ys)
            adv_loss = F.nll_loss(apreds, attrs)
            running_loss += loss.item()
            running_adv_loss += adv_loss.item()
            loss += mu * adv_loss
            loss.backward()
            optimizer.step()
        logger.info("Iteration {}, loss value = {}, adv_loss value = {}".format(t, running_loss, running_adv_loss))
    time_end = time.time()
    logger.info("Time used for training = {} seconds.".format(time_end - time_start))
    net.eval()
    preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
elif args.model == "cfair" or args.model == "cfair-eo":
    # Training with CFairNet to show the debiased results.
    logger.info("Experiment with CFairNet: {} adversarial debiasing:".format(args.model))
    logger.info("Hyperparameter setting = {}.".format(configs))
    time_start = time.time()
    net = CFairNet(configs).to(device)
    logger.info("Model architecture: {}".format(net))
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    mu = args.mu
    net.train()
    for t in range(num_epochs):
        running_loss, running_adv_loss = 0.0, 0.0
        for xs, ys, attrs in train_loader:
            xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
            optimizer.zero_grad()
            ypreds, apreds = net(xs, ys)
            # Compute both the prediction loss and the adversarial loss. Note that in CFairNet, both are conditional
            # losses.
            loss = F.nll_loss(ypreds, ys, weight=reweight_target_tensor)
            adv_loss = torch.mean(torch.stack([F.nll_loss(apreds[j], attrs[ys == j], weight=reweight_attr_tensors[j])
                                              for j in range(num_classes)]))
            running_loss += loss.item()
            running_adv_loss += adv_loss.item()
            loss += mu * adv_loss
            loss.backward()
            optimizer.step()
        logger.info("Iteration {}, loss value = {}, adv_loss value = {}".format(t, running_loss, running_adv_loss))
    time_end = time.time()
    logger.info("Time used for training = {} seconds.".format(time_end - time_start))
    net.eval()
    preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
else:
    raise NotImplementedError("{} not supported.".format(args.model))
# Print out all the statistics.
logger.info("The global marginal label distribution of Y = 1: {}".format(label_marginal))
logger.info("Overall predicted error = {}, Err|A=0 = {}, Err|A=1 = {}".format(cls_error, error_0, error_1))
logger.info("Joint Error: |Err|A=0 + Err|A=1| = {}".format(error_0 + error_1))
logger.info("Error Gap: |Err|A=0 - Err|A=1| = {}".format(np.abs(error_0 - error_1)))
logger.info("DP Gap: |Pred=1|A=0 - Pred=1|A=1| = {}".format(np.abs(pred_0 - pred_1)))
logger.info("Equalized Odds Y = 0: |Pred = 1|A = 0, Y = 0 - Pred = 1|A = 1, Y = 0| = {}".format(
    np.abs(cond_00 - cond_10)))
logger.info("Equalized Odds Y = 1: |Pred = 1|A = 0, Y = 1 - Pred = 1|A = 1, Y = 1| = {}".format(
    np.abs(cond_01 - cond_11)))
logger.info("*" * 50)
logger.info("Pr(Y = 1|A = 0) = {}".format(base_0))
logger.info("Pr(Y = 1|A = 1) = {}".format(base_1))
logger.info("Total Variation Lower bound = {}".format(np.abs(base_0 - base_1)))
logger.info("*" * 100)

# Save all the results.
out_file = "adult_{}_{}.npz".format(args.model, args.mu)
np.savez(out_file, prediction=preds_labels, truth=target_labels, attribute=target_attrs)
