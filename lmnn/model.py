#!/usr/bin/env python3
from sklearn.model_selection import train_test_split

from lmnn.layers.dropout import DropoutLayer
from .layers.struct import LayerStruct
from .loss.struct import LossStruct
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import math

class lmnn():
    def __init__(self, layers:list[LayerStruct], loss:LossStruct, lr=0.1, n_iter=1000, test_size=0.2, strategy="full", sub_parts=5, patience=50):
        self.lr = lr
        self.n_iter = n_iter
        self.test_size = test_size
        self.layers = layers
        self.layers.insert(0, None)
        self.strategy = strategy
        self.sub_parts = sub_parts
        self.nb_layers = len(layers)
        self.training_history = np.zeros((int(self.n_iter), 3))
        self.patience = patience
        self.loss = loss
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def forward_propagation(self, current_X, predicting=False):
        activations = {'A0' : current_X}
        previous_layer_index = 0

        for c in range(1, self.nb_layers):
            if predicting == True and isinstance(self.layers[c], DropoutLayer):
                continue
            activations['A' + str(c)] = self.layers[c].activate(activations['A' + str(previous_layer_index)])
            previous_layer_index = c

        return activations
    
    def back_propagation(self, y, activations):
        m = y.shape[1]
        dL = self.loss.dl(activations['A' + str(self.nb_layers - 1)], y)
        gradients = {}

        for c in reversed(range(1, self.nb_layers)):
            dZ = dL * self.layers[c].da(activations['A' + str(c)])
            gradients['dW' + str(c)] = self.layers[c].dw(m, dZ, activations['A' + str(c - 1)])
            gradients['db' + str(c)] = self.layers[c].db(m, dZ)
            if c > 1:
                dL = self.layers[c].dz(dZ, activations['A' + str(c - 1)])

        return gradients
    
    def update(self, gradients):
        for c in range(1, self.nb_layers):
            self.layers[c].update(gradients['dW' + str(c)], gradients['db' + str(c)], self.lr)
        return
    
    def predict(self, X):
        activations = self.forward_propagation(X, predicting=True)
        Af = activations['A' + str(self.nb_layers - 1)]
        return Af >= 0.5
    
    def save_results(self, i, Af, X_train, y_train):
        # calcul du log_loss et de l'accuracy
        test = self.loss.compute_loss(Af, y_train)
        self.training_history[i, 0] = np.mean(test)
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(self.X_test)
        self.training_history[i, 1] = (accuracy_score(np.argmax(y_train, axis=0), np.argmax(y_pred_train, axis=0)))
        self.training_history[i, 2] = (accuracy_score(np.argmax(self.y_test, axis=0), np.argmax(y_pred_test, axis=0)))

    def fit(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        if(self.strategy == "full"):
            for i in tqdm(range(self.n_iter)):
                activations = self.forward_propagation(self.X_train)
                gradients = self.back_propagation(self.y_train, activations)
                self.update(gradients)
                Af = activations['A' + str(self.nb_layers - 1)]
                self.save_results(i, Af, self.X_train, self.y_train)
                if i % self.patience == 0 and ((self.training_history[i, 1] - self.training_history[i - self.patience, 1]) < 0.0001):
                    break

        elif(self.strategy == "sub"):
            #Not corrected 
            self.n_iter = (np.floor(self.n_iter / self.sub_parts)).astype(int)
            portion = 1 / self.sub_parts
            nb_element_sub_train = math.floor(self.X_train.shape[1] * portion)
            
            e = 0
            early_stop = False
            for i in tqdm(range(self.n_iter -1 )):
                for x in range(0, self.sub_parts):
                    e+=1
                    start_train_index = nb_element_sub_train * x
                    end_train_index = nb_element_sub_train * (x+1)
                    X_train_sub = self.X_train[:, start_train_index:end_train_index]
                    y_train_sub = self.y_train[:, start_train_index:end_train_index]

                    activations = self.forward_propagation(X_train_sub)
                    gradients = self.back_propagation(y_train_sub, activations)
                    self.update(gradients)
                    Af = activations['A' + str(self.nb_layers - 1)]
                    self.save_results(e, Af, X_train_sub, y_train_sub)
                    if e % self.patience == 0 and ((self.training_history[e, 1] - self.training_history[e - self.patience, 1]) < 0.0001):
                        early_stop = True
                        break
                if early_stop == True:
                    break
        else:
            raise ValueError("Unsupported strategy")