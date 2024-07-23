#!/usr/bin/env python3
# Import should be able to retreive all activations more conviniently
from lmnn.gpu.layers.structures import DenseLayer, DropoutLayer, OutputLayer, LayerStruct
from lmnn.gpu.activations.functions import SigmoidActivation


dl =  DenseLayer(SigmoidActivation(), 15)

test = dl.activate([1,5,6,8,9])