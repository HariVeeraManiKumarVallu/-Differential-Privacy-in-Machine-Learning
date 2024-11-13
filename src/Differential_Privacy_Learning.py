import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer as DPKerasSGDOptimizer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

def create_model():
    """
    Create and compile a neural network model with differential privacy.
    """
    keras_model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),  # Increased units for better learning capacity
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=0.1,
        noise_multiplier=0.1,
        num_microbatches=1,  # Set to match batch_size for compatibility
        learning_rate=0.01
    )

    keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return keras_model

def train_model():
    """
    Load the MNIST dataset, preprocess it, train the model with differential privacy,
    and return the training history.
    """
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data: Flatten and normalize correctly
    x_train = x_train.reshape((60000, 784)).astype('float32') / 255  # Correct normalization
    x_test = x_test.reshape((10000, 784)).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Create and train the model
    keras_model = create_model()
    
    # Early stopping with reduced patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    history = keras_model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.02,
        verbose=1,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    print("Model training complete.")
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = keras_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return history

# Run the training function
if __name__ == "__main__":
    history = train_model()
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()