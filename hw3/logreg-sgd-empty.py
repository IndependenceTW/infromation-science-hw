#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import copy
import math
import sys

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header=None,
                           names=['x%i' % (i) for i in range(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])
    X = numpy.hstack((numpy.ones((X.shape[0], 1)), X))
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(
        feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss


def logreg_sgd(X, y, alpha=.001, epochs=10000, eps=1e-4):
    # TODO: compute theta
    # alpha: step size
    # epochs: max epochs
    # eps: stop when the thetas between two epochs are all less than eps
    n, d = X.shape
    theta = numpy.zeros((d, 1))

    # Start Implement

    # repeat no more the max epochs
    for _ in range(epochs):
        # do gradient ascend to each X
        pre_theta = theta

        for i in range(n):
            # diff = X^T * (y - y_predict)
            # ∆ = ɑ * dif
            # theta^(k+1) = theta^k + ∆
            y_predict = predict_prob(X[i], theta).item()
            diff = X[i].reshape((d, 1)) * (y[i] - y_predict)
            delta = alpha * diff
            theta = theta + delta

        print(theta)

        dif_between_theta = numpy.abs(pre_theta - theta)
        counter = 0
        for i in range(d):
            if dif_between_theta[i].item() < eps:
                counter += 1

        if (counter == d):
            break

    # End

    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []

    # Start Implement
    # 1. counter the number of true(1) and false(0)
    false_num = 0
    true_num = 0

    for y_value in y_test:
        if y_value == 0:
            false_num += 1
        if y_value == 1:
            true_num += 1

    # 1.5 merge the y_test_data and y_prob
    y_dataset = []
    for i in range(len(y_test)):
        y_dataset.append([y_prob[i], y_test[i]])

    # 2. sort the data with y_prob
    y_dataset.sort(key=lambda x: x[0], reverse=True)

    # 3. count tpr and fpr
    now_tpr = 0
    now_fpr = 0
    for y_data in y_dataset:
        if (y_data[1] == 0):
            now_fpr += (1. / false_num)
        if (y_data[1] == 1):
            now_tpr += (1. / true_num)
        tpr.append(now_tpr)
        fpr.append(now_fpr)

    # End

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" %
          (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    print("Logreg train precision: %f" %
          (sklearn.metrics.precision_score(y_train, y_prob > .5)))
    print("Logreg train recall: %f" %
          (sklearn.metrics.recall_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" %
          (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    print("Logreg test precision: %f" %
          (sklearn.metrics.precision_score(y_test, y_prob > .5)))
    print("Logreg test recall: %f" %
          (sklearn.metrics.recall_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)
