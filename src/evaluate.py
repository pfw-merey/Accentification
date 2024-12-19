import numpy as np
from data_loader import load_data
from feature_extractor import extract_features
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
DATA_PATH = "data"
MODEL_PATH = "models/v1_balanced.keras"

# Load and preprocess data
features, labels = load_data(DATA_PATH)
features = np.array([extract_features(f) for f in features])
labels = np.array(labels)

# Split data into training and testing sets
_, X_test, _, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape features to add sequence dimension for Conv1D
X_test = X_test[..., np.newaxis]  # Shape: (batch_size, num_features, 1)

# Load the trained model
model = load_model(MODEL_PATH)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Generate classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["American", "British"]))
