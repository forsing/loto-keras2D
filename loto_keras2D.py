"""        
=== System Information ===
Python version                 3.11.13        
macOS Apple                    Tahos 
Apple                          M1
"""


"""
bash

pip uninstall -y keras keras-core keras-tuner tensorflow tensorflow-macos tensorflow-metal
pip install tensorflow-macos keras
python3 -c "from keras.models import Sequential; print('✅ Radi!')"
"""


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4506 izvlacenja
30.07.1985.- 04.11.2025.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import tensorflow as tf

import os
import random

# -----------------------------
# 1️⃣ Set seed for reproducibility
# -----------------------------
SEED = 39
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 1️⃣ Load and preprocess data
# -----------------------------
df = pd.read_csv('/data/loto7_4506_k87.csv', header=None)


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Split into train and test
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# -----------------------------
# 2️⃣ Create dataset with look_back
# -----------------------------
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:i+look_back].flatten()  # flatten sequence
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# -----------------------------
# 3️⃣ Build model
# -----------------------------
input_dim = trainX.shape[1]  # look_back * 7
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='linear'))  # linear for regression

model.compile(
    loss='mean_squared_error',
    optimizer=optimizers.Adam(learning_rate=0.0002),
    metrics=['mae']
)

# -----------------------------
# 4️⃣ Train model
# -----------------------------
model.fit(trainX, trainY, epochs=1000, batch_size=50, verbose=1)

# -----------------------------
# 5️⃣ Multi-prediction for stability
# -----------------------------
predictions = []
for _ in range(1000):  # 10 predictions
    pred_scaled = model.predict(testX[0].reshape(1, -1))
    pred = scaler.inverse_transform(pred_scaled)
    pred = np.round(pred).astype(int)[0]
    predictions.append(pred)

# Aggregate predictions (mode of each number)
predicted_numbers = [int(np.bincount([p[i] for p in predictions]).argmax()) for i in range(7)]

# -----------------------------
# 6️⃣ Show final predicted numbers
# -----------------------------
print()
print(f"The predicted next set of numbers is: {predicted_numbers}")
print()
"""
The predicted next set of numbers is: [4 9 x x x 29 35]
"""


 
