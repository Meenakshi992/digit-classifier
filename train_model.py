# Part-1 
# Trains & saves 'mnist_cnn_enhanced.h5'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Load and normalize dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 2. Build the CNN Architecture with BN and Dropout
model = models.Sequential([
    # First Conv Block
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Conv Block
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(256),  # Increased capacity for richer feature matching
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

# 3. Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training high-stability CNN with Batch Normalization...")
model.fit(x_train, y_train, epochs=6, batch_size=64, validation_split=0.1)

# 4. Export model
model.save("mnist_cnn_enhanced.h5")
print("Model saved successfully as 'mnist_cnn_enhanced.h5'!")
