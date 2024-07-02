#!/usr/bin/env python3
from ..layers.dense import DenseLayer

dl =  DenseLayer(15, "relu")

dl.activate()

print(dl.activation)
print(dl.nb_neurons)