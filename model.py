import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

def train_model(data):
    # Split data into features and labels
    X = data[:, :-1]
    y = data[:, -1]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build and compile the TensorFlow model
    model = tf.keras.Sequential([
        # Define model layers
        # ...
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    # Save the trained model
    model.save('models/trained_model.h5')

def predict_moisture(data):
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    normalized_data = scaler.transform(data)
    model = tf.keras.models.load_model('models/trained_model.h5')
    predictions = model.predict(normalized_data)
    predicted_data = np.hstack((data, predictions.reshape(-1, 1)))
    return predicted_data

