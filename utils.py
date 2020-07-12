#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def conditional_errors(preds, labels, attrs):
    """
    Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 0, error | A = 1.
    """
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean(preds == labels)
    idx = attrs == 0
    error_0 = 1 - np.mean(preds[idx] == labels[idx])
    error_1 = 1 - np.mean(preds[~idx] == labels[~idx])
    return cls_error, error_0, error_1
