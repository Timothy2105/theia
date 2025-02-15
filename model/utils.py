import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_esp32_csi(csi_data, window_size=50):
    """
    Preprocess ESP32 CSI data
    
    Args:
        csi_data: Raw CSI amplitude data (n_samples, n_subcarriers)
                 Using 128 subcarriers for ESP32
        window_size: Number of packets per capture (default: 50)
    
    Returns:
        Processed features ready for neural network
    """
    n_samples = len(csi_data)
    windows = []
    
    for i in range(0, n_samples - window_size + 1):
        window = csi_data[i:i + window_size]
        features = []
        # Mean of each subcarrier
        features.extend(np.mean(window, axis=0))
        # Variance of each subcarrier
        features.extend(np.var(window, axis=0))
        # Max-Min range for each subcarrier
        features.extend(np.max(window, axis=0) - np.min(window, axis=0))
        
        windows.append(features)
    
    return np.array(windows)

class ESP32CSIMultiTaskModel:
    def __init__(self, input_shape=(128, 50)):
        """
        Initialize ESP32 CSI Model for both presence detection and location prediction
        
        Args:
            input_shape: Tuple of (n_subcarriers, window_size)
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural network architecture for multi-task learning"""
        # Input layer
        inputs = layers.Input(shape=(*self.input_shape, 1))
        
        # Shared layers
        x = layers.Conv2D(16, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2, strides=2)(x)
        
        x = layers.Conv2D(32, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2, strides=2)(x)
        
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
        
        shared_features = layers.Flatten()(x)
        
        # Presence detection branch
        presence_hidden = layers.Dense(32, activation='relu')(shared_features)
        presence_output = layers.Dense(1, activation='sigmoid', name='presence')(presence_hidden)
        
        # Location prediction branch
        location_hidden = layers.Dense(64, activation='relu')(shared_features)
        location_hidden = layers.Dense(32, activation='relu')(location_hidden)
        location_output = layers.Dense(2, name='location')(location_hidden)
        
        # Create model with multiple outputs
        model = models.Model(
            inputs=inputs,
            outputs=[presence_output, location_output]
        )
        
        # Custom loss for location prediction that only applies when person is present
        def masked_mse(y_true, y_pred):
            # Convert inputs to float32
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            presence_mask = y_true[:, 0] != -1
            squared_error = tf.square(y_true - y_pred)
            mse = tf.reduce_mean(squared_error, axis=-1)
            masked_mse = tf.reduce_mean(tf.boolean_mask(mse, presence_mask))
            return masked_mse
        
        # Custom metric for location prediction accuracy
        def location_mae(y_true, y_pred):
            # Convert inputs to float32
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            presence_mask = y_true[:, 0] != -1
            absolute_error = tf.abs(y_true - y_pred)
            mae = tf.reduce_mean(absolute_error, axis=-1)
            return tf.reduce_mean(tf.boolean_mask(mae, presence_mask))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'presence': 'binary_crossentropy',
                'location': masked_mse
            },
            metrics={
                'presence': ['accuracy'],
                'location': [location_mae]
            },
            loss_weights={
                'presence': 1.0,
                'location': 0.5  # Adjust this weight to balance the tasks
            }
        )
        
        return model

def parse_csi_data(csi_string):
    """
    Parse CSI data from string format to numpy array
    
    Args:
        csi_string: String containing CSI data in array format
        
    Returns:
        numpy array of CSI values
    """
    # Remove brackets and split into values
    values = csi_string.strip('[]').split()
    # Convert to float array
    return np.array([float(x) for x in values])

def prepare_esp32_data_multitask(csv_file, window_size=50):
    """
    Prepare ESP32 CSI data for multi-task learning
    
    Args:
        csv_file: Path to CSV file containing CSI data
        window_size: Number of packets per window
    
    Returns:
        Processed data ready for training
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract CSI data from the CSV
    csi_data = df['CSI_DATA'].apply(parse_csi_data).values
    csi_data = np.stack(csi_data)
    
    # Extract labels
    presence_labels = df['state'].values
    locations = df[['locationX', 'locationY']].values
    
    # Reshape CSI data into windows
    n_samples = len(csi_data) // window_size
    csi_windows = csi_data[:n_samples * window_size].reshape(n_samples, -1, window_size)
    
    # Add channel dimension for Conv2D
    csi_windows = csi_windows[..., np.newaxis]
    
    # Prepare corresponding labels for windows
    # Take the most common presence value in each window
    presence_windows = np.array([
        np.bincount(presence_labels[i:i+window_size]).argmax()
        for i in range(0, n_samples * window_size, window_size)
    ])
    
    # Take the mean of valid locations in each window
    location_windows = np.array([
        np.mean(locations[i:i+window_size], axis=0)
        if presence_windows[i//window_size] == 1
        else np.array([-1, -1])
        for i in range(0, n_samples * window_size, window_size)
    ])
    
    # Convert data types to float32
    csi_windows = csi_windows.astype(np.float32)
    presence_windows = presence_windows.astype(np.float32)
    location_windows = location_windows.astype(np.float32)
    
    return train_test_split(
        csi_windows,
        presence_windows,
        location_windows,
        test_size=0.2,
        random_state=42
    )

def train_esp32_multitask_model(csv_file, input_shape=(128, 50), epochs=10, batch_size=8):
    """
    Train the ESP32 CSI multi-task model
    
    Args:
        csv_file: Path to CSV file containing CSI data
        input_shape: Shape of input data (n_subcarriers, window_size)
        epochs: Number of training epochs
        batch_size: Size of training batches
    
    Returns:
        Trained model and training history
    """
    # Prepare data
    X_train, X_val, y_train_presence, y_val_presence, y_train_location, y_val_location = \
        prepare_esp32_data_multitask(csv_file)
    
    # Initialize model
    model = ESP32CSIMultiTaskModel(input_shape=input_shape)
    
    # Train model
    history = model.model.fit(
        X_train,
        {
            'presence': y_train_presence,
            'location': y_train_location
        },
        validation_data=(
            X_val,
            {
                'presence': y_val_presence,
                'location': y_val_location
            }
        ),
        epochs=epochs,
        batch_size=batch_size,
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
    """
    Plot training metrics for multi-task model
    
    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(15, 5))
    
    # Plot presence detection metrics
    plt.subplot(1, 3, 1)
    plt.plot(history.history['presence_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_presence_accuracy'], label='Validation Accuracy')
    plt.title('Presence Detection Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot presence detection loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['presence_loss'], label='Training Loss')
    plt.plot(history.history['val_presence_loss'], label='Validation Loss')
    plt.title('Presence Detection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot location prediction loss
    plt.subplot(1, 3, 3)
    plt.plot(history.history['location_loss'], label='Training MAE')
    plt.plot(history.history['val_location_loss'], label='Validation MAE')
    plt.title('Location Prediction MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_test, y_test_presence, y_test_location, num_samples=5):
    """
    Visualize model predictions
    
    Args:
        model: Trained ESP32CSIMultiTaskModel
        X_test: Test input data
        y_test_presence: True presence labels
        y_test_location: True location coordinates
        num_samples: Number of samples to visualize
    """
    # Get predictions
    presence_pred, location_pred = model.model.predict(X_test[:num_samples])
    
    # Print results
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        print(f"Presence: True={y_test_presence[i]}, Predicted={presence_pred[i][0]:.2f}")
        if y_test_presence[i] == 1:
            print(f"Location: True=({y_test_location[i][0]:.1f}, {y_test_location[i][1]:.1f}), "
                  f"Predicted=({location_pred[i][0]:.1f}, {location_pred[i][1]:.1f})")