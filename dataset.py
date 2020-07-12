#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_logger

VAR_IND = {'income': 'binary_catergorial',
           'sex': 'binary_catergorial'}

logger = logging.getLogger(__name__)


class COMPAS(Dataset):
    """
    The COMPAS dataset.
    """
    def __init__(self, data):
        """
        :param data:    Numpy array that contains all the data.
        """
        # Data, label and sensitive attribute partition.
        self.insts = data[:, 1:].astype(np.float32)
        self.labels = data[:, 0].astype(np.int64)
        self.attrs = data[:, 5].astype(np.int64)
        self.xdim = self.insts.shape[1]

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return torch.tensor(self.insts[idx]), \
               torch.tensor(self.labels[idx]), \
               torch.tensor(self.attrs[idx])


class AdultDataset(Dataset):
    """
    The UCI Adult dataset.
    """

    def __init__(self, root_dir, phase, tar_attr, priv_attr):
        self.tar_attr = tar_attr
        self.priv_attr = priv_attr
        self.npz_file = os.path.join(root_dir, 'adult_%s_%s.npz' % (tar_attr, priv_attr))
        self.data = np.load(self.npz_file)
        if phase == "train":
            self.X = self.data["x_train"]
            self.Y = self.data["y_train"]
            self.A = self.data["attr_train"]
        elif phase == "test":
            self.X = self.data["x_test"]
            self.Y = self.data["y_test"]
            self.A = self.data["attr_test"]
        else:
            raise NotImplementedError

        self.xdim = self.X.shape[1]
        self.ydim = self.Y.shape[1] if self.Y.shape[1] != 2 else 1
        self.adim = self.A.shape[1] if self.Y.shape[1] != 2 else 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.ydim == 1 and len(self.Y.shape) == 2:  # binary classification
            return torch.from_numpy(self.X[idx]).float(), \
                   self.onehot_2_int(torch.from_numpy(self.Y[idx])), \
                   self.onehot_2_int(torch.from_numpy(self.A[idx]))
        raise NotImplementedError

    def onehot_2_int(self, ts):
        if len(ts.shape) == 2:
            return torch.argmax(ts, dim=1)
        if len(ts.shape) == 1:
            return torch.argmax(ts, dim=0)
        raise NotImplementedError

    def get_A_proportions(self):
        """ for catergorical attribute
        """
        assert len(self.A.shape) == 2
        num_class = self.A.shape[1]

        A_label = np.argmax(self.A, axis=1)
        A_proportions = []
        for cls_idx in range(num_class):
            A_proportion = np.sum(cls_idx == A_label)
            A_proportions.append(A_proportion)
        A_proportions = [a_prop * 1.0 / len(A_label) for a_prop in A_proportions]
        return A_proportions

    def get_Y_proportions(self):
        """ for catergorical attribute
        """
        assert len(self.Y.shape) == 2
        num_class = self.Y.shape[1]

        Y_label = np.argmax(self.Y, axis=1)
        Y_proportions = []
        for cls_idx in range(num_class):
            Y_proportion = np.sum(cls_idx == Y_label)
            Y_proportions.append(Y_proportion)
        Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
        return Y_proportions

    def get_AY_proportions(self):
        """ for catergorical attributes
        """
        assert len(self.Y.shape) == len(self.A.shape) == 2
        A_num_class = self.A.shape[1]
        Y_num_class = self.Y.shape[1]
        A_label = np.argmax(self.A, axis=1)
        Y_label = np.argmax(self.Y, axis=1)
        AY_proportions = []
        for A_cls_idx in range(A_num_class):
            Y_proportions = []
            for Y_cls_idx in range(Y_num_class):
                AY_proprtion = np.sum(np.logical_and(Y_cls_idx == Y_label, A_cls_idx == A_label))
                Y_proportions.append(AY_proprtion)
            Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
            AY_proportions.append(Y_proportions)
        return AY_proportions


if __name__ == '__main__':
    logger = get_logger("dataset")
    trainset = AdultDataset(root_dir='data', phase='train', tar_attr='income', priv_attr='sex')
    testset = AdultDataset(root_dir='data', phase='test', tar_attr='income', priv_attr='sex')

    print(len(trainset), trainset[:100][0].shape, trainset[:100][1].shape, trainset[:100][2].shape)
    print(len(testset), testset[0][0].shape, testset[:100][0].shape, testset[:100][1].shape,
          testset[:100][2].shape)
    print(trainset[:100][0].dtype, trainset[:100][1].dtype, trainset[:100][2].dtype)

    logger.info("Train Label: {}".format(trainset.get_Y_proportions()))
    logger.info("Test Label: {}".format(testset.get_Y_proportions()))

    logger.info("Train Attr: {}".format(trainset.get_A_proportions()))
    logger.info("Test Attr: {}".format(testset.get_A_proportions()))

    logger.info("Train Label + Attr: {}".format(trainset.get_AY_proportions()))
    logger.info("Test Label + Attr: {}".format(testset.get_AY_proportions()))
