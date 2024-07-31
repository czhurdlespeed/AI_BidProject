import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

X_data = np.load('X_data.npy', allow_pickle=True)
y_data = np.load('y_data.npy', allow_pickle=True)

scaler_X = StandardScaler()
X_data[:, -1] = scaler_X.fit_transform(X_data[:, -1].reshape(-1, 1)).flatten()


# Scale y_data[:, 1] (assuming this is a continuous variable you want to scale)
scaler_y = StandardScaler()
y_data[:, 1] = scaler_y.fit_transform(y_data[:, 1].reshape(-1, 1)).flatten()

print(X_data.shape)
print(y_data.shape)
print(X_data)
print(y_data)

y_binary = y_data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_binary, test_size=0.1, random_state=42, stratify=y_binary)

# Convert to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int).flatten() 

print(f"Y test: {y_test[:10]}")
print(f"Y pred: {y_pred[:10]}")

print(classification_report(y_test, y_pred_classes))


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
