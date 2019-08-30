#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Proof of concept of a minimalist, 3 layers (1 hidden) Neural Network. Code
based on the lectures from the Machine Learning coursework (Stanford -
Coursera) by Prof. Dr. Andrew Ng and the code implementation of andr0idsensei
(https://gist.github.com/andr0idsensei/92dd7e54a029242690555e84dca93efd).

@author: Dr. Matheus Boni Vicari
"""

from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    
    # Setting up parameters.
    hidden_size = 30
    num_labels = 10
    learning_rate = 0.2
    J_threshold = 0.0001
    regularize = True
    n_iter = 8000

    # Loading training data.
    data = loadmat('data/mnist_sample.mat')
    X = np.matrix(data['X'])
    y = np.matrix(data['y'])    
    m = X.shape[0]

    # Initializing random parameters (weights) for the activation of 
    # layers 1 and 2.
    e_init = np.sqrt(6) / np.sqrt(X.shape[1] + hidden_size)
    theta1 = np.random.rand(hidden_size, X.shape[1] + 1) * (2 * e_init) - e_init
    e_init = np.sqrt(6) / np.sqrt(num_labels + hidden_size)
    theta2 = np.random.rand(num_labels, hidden_size + 1) * (2 * e_init) - e_init
    
    # One Hot encoder.
    y_encoded = np.zeros([m, num_labels], dtype=int)
    y_encoded[np.arange(m), np.ravel(y) - 1] = 1
       
    # Running optimization with simple gradient descent. Loop will run
    # until max_iter is reached or the difference between consecutive cost
    # functions J drop below J_threshold.
    J_prev = np.inf
    for i in xrange(n_iter):
        ## FORWARD PROPAGATION
        # Inserting a column of ones as bias units to original X.
        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        # Calculating hypothesis for hidden layer.
        z2 = a1 * theta1.T  
        hz2 = 1 / (1 + np.exp((-z2)))
        # Inserting a column of ones as bias units to hidden layer.
        a2 = np.insert(hz2, 0, values=np.ones(m), axis=1)
        # Calculating hypothesis for output layer.
        h = 1 / (1 + np.exp((-a2 * theta2.T)))
        
        # Calculating cost function
        J = (np.multiply(-y_encoded, np.log(h)) -
             np.multiply((1 - y_encoded), np.log(1 - h))).sum() / m

        # Regularizing cost function if 'regularization' is set to True.
        if regularize:
            J += (float(learning_rate) /
                  (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) +
                  np.sum(np.power(theta2[:, 1:], 2)))

        ## BACK PROPAGATION
        # Calculates difference (d) between prediction (h) and reference
        # (y_encoded).
        d3 = h - y_encoded
        
        # Inserting a column of ones as bias units to hidden layer.
        z2 = np.insert(z2, 0, values=np.ones(1), axis=1)
        # Calculating d for hidden layer.
        hz2 = 1 / (1 + np.exp((-z2)))
        hg2 = np.multiply(hz2, (1 - hz2))
        d2 = np.multiply((theta2.T * d3.T).T, hg2)
        
        # Calculating deltas for hidden and output layers.
        delta1 = ((d2[:, 1:]).T * a1) / m
        delta2 = (d3.T * a2) / m

        # Regularizing deltas if required.
        if regularize:
            delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
            delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

        # Updating weights.
        theta1 -= delta1
        theta2 -= delta2

        # Calculating difference in cost function, checking against
        # threshold condition and updanting J_prev.
        if np.abs(J_prev - J) <= J_threshold:
            break
        J_prev = J
            
    ## PREDICTION
    # Inserting a column of ones as bias units to original X.
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    # Calculating hypothesis for hidden layer.
    hz2 = 1 / (1 + np.exp((-a1 * theta1.T)))
    # Inserting a column of ones as bias units to hidden layer.
    a2 = np.insert(hz2, 0, values=np.ones(m), axis=1)
    # Calculating hypothesis for output layer.
    h = 1 / (1 + np.exp((-a2 * theta2.T)))
    # Generating predictions.
    y_pred = np.array(np.argmax(h, axis=1) + 1)

    # Calculating performance metrics.
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('Accuracy = {0}'.format(accuracy)) 
