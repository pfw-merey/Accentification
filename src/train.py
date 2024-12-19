import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data_loader import load_data, oversample_data, augment_audio
from feature_extractor import extract_features
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = "data"
BATCH_SIZE = 32
EPOCHS = 50

# Load and preprocess data
features, labels = load_data(DATA_PATH)

# Apply oversampling to balance the dataset
target_label = 'british'
target_count = len([l for l in labels if l == 'american'])
features, labels = oversample_data(features, labels, target_count, target_label)

# Apply data augmentation
augmented_features = []
augmented_labels = []
for audio, label in zip(features, labels):
    augmented_features.extend([audio] + augment_audio(audio))
    augmented_labels.extend([label] * (1 + len(augment_audio(audio))))

features, labels = augmented_features, augmented_labels

# Extract features for all audio samples
features = np.array([extract_features(f) for f in features])
labels = np.array(labels)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape features to add sequence dimension for Conv1D
X_train = X_train[..., np.newaxis]  # Shape: (batch_size, num_features, 1)
X_test = X_test[..., np.newaxis]

# Compute class weights to handle remaining imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Build the model
input_shape = X_train.shape[1:]
model = build_model(input_shape)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save('models/v1_balanced.keras')
