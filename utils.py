import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Core functionality classes and functions
def preprocess_esp32_csi(csi_data, window_size=50):
    """
    Preprocess ESP32 CSI data
    
    Args:
        csi_data: Raw CSI amplitude data (n_samples, n_subcarriers)
                  Using 30 subcarriers for ESP32
        window_size: Number of packets per capture (default: 50)
    
    Returns:
        Processed features ready for neural network
    """
    n_samples = len(csi_data)
    n_features = csi_data.shape[1]
    windows = []
    
    # Create sliding windows
    for i in range(0, n_samples - window_size + 1):
        window = csi_data[i:i + window_size]
        
        # Extract features from window
        features = []
        # Mean of each subcarrier
        features.extend(np.mean(window, axis=0))
        # Variance of each subcarrier
        features.extend(np.var(window, axis=0))
        # Max-Min range for each subcarrier
        features.extend(np.max(window, axis=0) - np.min(window, axis=0))
        
        windows.append(features)
    
    return np.array(windows)

class ESP32CSIModel:
    def __init__(self, input_shape=(30, 50)):
        """
        Initialize ESP32 CSI Model
        
        Args:
            input_shape: Tuple of (n_subcarriers, window_size)
                         (30, 50) for ESP32 data
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural network architecture adapted for ESP32 CSI data"""
        model = models.Sequential([
            layers.Input(shape=(*self.input_shape, 1)),
            
            layers.Conv2D(4, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, strides=2),
            
            layers.Conv2D(8, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, strides=2),
            
            layers.Conv2D(16, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.1),
            
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def prepare_esp32_data(csi_data, labels, train_ratio=0.8):
    """
    Prepare ESP32 CSI data for training
    
    Args:
        csi_data: Raw CSI data from ESP32 (n_samples, 30, 50)
        labels: Binary labels (0: absent, 1: present)
        train_ratio: Ratio of training data
    
    Returns:
        Processed data ready for training
    """
    if len(csi_data.shape) == 2:
        n_samples = csi_data.shape[0] // 50  # Using 50 packets per capture
        csi_data = csi_data[:n_samples * 50].reshape(n_samples, -1, 50)
    
    csi_data = csi_data[..., np.newaxis]
    
    X_train, X_val, y_train, y_val = train_test_split(
        csi_data, labels[:len(csi_data)],
        train_size=train_ratio,
        random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val)

def train_esp32_model(csi_data, labels, input_shape=(30, 50), epochs=10, batch_size=8):
    """
    Train the ESP32 CSI classifier
    
    Args:
        csi_data: Raw CSI data from ESP32
        labels: Binary labels
        input_shape: Shape of input data (n_subcarriers, window_size)
        epochs: Number of training epochs
        batch_size: Size of training batches
    
    Returns:
        Trained model and training history
    """
    (X_train, y_train), (X_val, y_val) = prepare_esp32_data(
        csi_data, labels
    )
    
    model = ESP32CSIModel(input_shape=input_shape)
    
    history = model.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )
        ]
    )
    
    return model, history

def plot_training_results(history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()