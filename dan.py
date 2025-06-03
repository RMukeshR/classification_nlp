import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, Adagrad
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("data/text_embedding.csv")

def convert_embedding(embedding_str):
    embedding_list = embedding_str.strip('[]').split()
    return np.array(embedding_list, dtype=np.float32)


df['w2v_embedding'] = df['w2v_embedding'].apply(convert_embedding)
df['fasttest_embedding'] = df['fasttest_embedding'].apply(convert_embedding)

df['conbined_w2v_fat'] = df.apply(lambda row: np.concatenate([row['w2v_embedding'], row['fasttest_embedding']]), axis=1)

x = np.array(df['conbined_w2v_fat'].tolist())
y = df['level']
# print(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  




X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=69)


X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)



model = Sequential([
    Input(shape=(600,)), 
    Dense(300, activation='relu'),  
    Dropout(0.3),
    Dense(100, activation='relu'), 
    Dropout(0.3),
    # Dense(50, activation='relu'), 
    Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile(optimizer=Adagrad(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

