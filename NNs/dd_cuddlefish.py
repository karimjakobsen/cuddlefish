import deepdig
from deepdig.layers.activation import Sigmoid
from deepdig.layers.dense import Dense
from deepdig.models.sequential import Sequential
import numpy as np
import pandas as pd


# Load data
file_path = "C:\\Users\\Jakal\\cuddlefish\\data\\colors.csv"
data = pd.read_csv(file_path)

# features (target color)
X = data[['r','b','g']]

# scale value in each position by 255 (max rgb value = 255)
X = X.values / 255

# labels (base colors)
y = data[['red','yellow','blue','black','white']].values

color_sums = y.sum(axis=1).reshape(-1,1)

# convert to ratios
y = y / color_sums
#check shapes
print('X shape', np.shape(X))
print('X shape', np.shape(X[1]))
print('y shape', np.shape(y))

output_shape = np.shape(y)[1]
print(output_shape)

# Create a model: x -> sigmoid(x*w1+b1)=a1 -> sigmoid(a1*w2+b2)=a2 -> y

model = Sequential([Dense(neurons=100, activation='sigmoid'),
                    Dense(neurons=100, activation='sigmoid'),
                    Dense(neurons=100, activation='sigmoid'),
                    Dense(neurons=output_shape, activation='sigmoid')],
                   optimizer = 'gradient_descent',
                   learning_rate = 0.1,
                   loss = 'mse',
                   epochs = 200)

# Build the model
model.build()

# Train model

model.train(X, y)

# Inspect model objects
#print(model.loss)
#print(model.layers)
#print(model.optimizer)
#print(model.cache)
#print(model.learning_rate)
#print(model.epochs)





