#!/usr/bin/env python3
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from lmnn.activations.softmax import SoftMaxActivation
from ..layers.dropout import DropoutLayer
from ..layers.dense import DenseLayer
from ..layers.output import OutputLayer
from ..model import lmnn
from ..activations.sigmoid import SigmoidActivation
from ..activations.relu import ReluActivation
from ..loss.bce import BceLoss
from ..initializers.xavier import XavierInitializer
from ..initializers.random import RandomInitializer
from ..initializers.he import HeInitializer
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False )
y = np.array([int(num) for num in y])
#print(X.shape)
X = X[:6000]
y = y[:6000]
print(np.unique(y, return_counts=True))
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)
X_train = X_train.T
X_test = X_test.T
#print(y_train.shape)
#print(y_test)

# Création de la matrice identité
identity_matrix = np.eye(10)
y_train = identity_matrix[y_train[0]].T
y_test = identity_matrix[y_test[0]].T

layers = [
    DenseLayer(ReluActivation(), XavierInitializer(startegy="normal"), 64),
    DenseLayer(SigmoidActivation(), HeInitializer(), 64),
    DenseLayer(ReluActivation(), XavierInitializer(startegy="normal"), 64),
    OutputLayer(SoftMaxActivation(), XavierInitializer(startegy="normal"),  10)
]

model = lmnn(layers, BceLoss(), n_iter=2800, lr=0.01, patience=350, strategy="sub", sub_parts=2)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
model.fit(X_train, X_test, y_train, y_test)





y_pred = model.predict(X_train)

print(np.unique(np.argmax(y_train, axis=0), return_counts=True))
print(np.unique(np.argmax(y_pred, axis=0), return_counts=True))

ac = accuracy_score(np.argmax(y_train, axis=0),  np.argmax(y_pred, axis=0))
re = recall_score(np.argmax(y_train, axis=0),  np.argmax(y_pred, axis=0), average='weighted', zero_division=1)
pr = precision_score(np.argmax(y_train, axis=0),  np.argmax(y_pred, axis=0), average='weighted', zero_division=1)

bar_width=0.2
zoomx=(1000, 1800)
zoomy=(0.60,0.90)
plt.figure(figsize=(14, 6))
plt.title('Model scores')
plt.grid(True)
plt.bar(1, ac, width=bar_width, label='Accuracy Score', color='#D1FF38')
plt.bar(2, re, width=bar_width, label='Recall Score', color='#ACD32A')
plt.bar(3, pr, width=bar_width, label='Precision Score', color='#84A21F')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model.training_history[:, 0])
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('L')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(model.training_history[:, 1].shape[0]), model.training_history[:, 1], color=('green', 0.5), label='Training accuracy')
plt.plot(range(model.training_history[:, 2].shape[0]), model.training_history[:, 2], color=('orange', 0.5),  label='Test accuracy')
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


ax = plt.subplot(1, 1, 1)
plt.plot(range(model.training_history[:, 1].shape[0]), model.training_history[:, 1], color=('green', 0.5), label='Training accuracy')
plt.plot(range(model.training_history[:, 2].shape[0]), model.training_history[:, 2], color=('orange', 0.5), label='Test accuracy')
ax.set_ylim(zoomy)
ax.set_xlim(zoomx)
plt.title('Learning Curve (zoomed)')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

disp = ConfusionMatrixDisplay.from_predictions(np.argmax(y_train, axis=0), np.argmax(y_pred, axis=0))
disp.figure_.suptitle("Confusion Matrix")
plt.show()