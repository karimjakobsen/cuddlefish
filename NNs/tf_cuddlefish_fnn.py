# imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# Load data
file_path = "C:\\Users\\Jakal\\cuddlefish\\data\\colors.csv"
data = pd.read_csv(file_path)

# features
X = data[['r','g','b']]
# scale features to be between 0 and 1
X = X.values / 255

# labels
y = data[['red','yellow','blue','black','white']].values

# here should add epsilon to avoid division by zero but seems like pandas did it auomatically to zero-values

# sum each mix
mix_sums = y.sum(axis=1).reshape(-1, 1)

# convert each color to ratio of sum
y = y / mix_sums

print("X shape: ", X.shape)
print("y shape: ", y.shape)

# make sure no NAN or INF values
assert not np.any(np.isnan(X))
assert not np.any(np.isnan(y))
assert not np.any(np.isinf(X))
assert not np.any(np.isinf(y))


# splits
                                                # Omitted for lack of data
# ==================================================================================================== #
#X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# shapes of splits
#print(f"X Train: {X_train.shape}, X Val: {X_val.shape}, X Test: {X_test.shape}")
#print(f"y Train: {y_train.shape}, y Val: {y_val.shape}, y Test: {y_test.shape}")
# ==================================================================================================== #


# Design model
                                                # MODEL
# ==================================================================================================== #
model = Sequential([
    Dense(64, activation='relu', input_shape = (3,)), # layer_1 has 64 neurons # ReLU because max(0, x)
    Dense(64, activation='relu'),                     # layer_2 has 64 neurons # ReLU because max(0, x)
    keras.layers.Dense(5, activation='softmax')       # output has 5 neurons # softmax to produce probabilities
    ])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# 
# train model # NOTE: done on X and y and not X_train, y_train
model.fit(X, y, epochs = 1000, batch_size = 22)

def predict_mix(rgb: np.ndarray):
    """
    Takes an np.ndarray input [[INT, INT, INT]] between 0 and 255
    divides it by 255 and predicts the base color mix ratio for that color.
    """

    for channel in rgb[0]:
        if channel < 0 > 255:
            raise Exception("RGB values can not be below 0 or above 255")
    rgb = rgb / 255

    return np.round(model.predict(rgb), 2)


