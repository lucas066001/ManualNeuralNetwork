#!/usr/bin/env python3
from ..layers.dense import DenseLayer
from ..activations.sigmoid import SigmoidActivation

dl =  DenseLayer(SigmoidActivation(), 15)

test = dl.activate([1,5,6,8,9])

#print(dl.activation)
#print(dl.nb_neurons)
#print(test)