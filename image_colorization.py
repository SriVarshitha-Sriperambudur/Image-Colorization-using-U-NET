import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import cv2
import os

# Part 1: Training the U-Net Model for Image Colorization
def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (x_train, _), (x_test, _) = cifar10.load_data()
    
    # Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert to grayscale (L channel in Lab color space)
    x_train_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train])
    x_test_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test])
    
    # Reshape grayscale images to have 1 channel
    x_train_gray = x_train_gray.reshape(-1, 32, 32, 1)
    x_test_gray = x_test_gray.reshape(-1, 32, 32, 1)
    
    return (x_train_gray, x_train), (x_test_gray, x_test)

def build_unet_model():
    inputs = Input(shape=(32, 32, 1))
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    concat4 = Concatenate()([up4, conv2])
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(concat4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up5 = UpSampling2D(size=(2, 2))(conv4)
    concat5 = Concatenate()([up5, conv1])
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(concat5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    # Output layer (RGB: 3 channels)
    outputs = Conv2D(3, 1, activation='sigmoid', padding='same')(conv5)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model():
    (x_train_gray, x_train), (x_test_gray, x_test) = load_and_preprocess_data()
    model = build_unet_model()
    
    # Train the model
    history = model.fit(
        x_train_gray, x_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_test_gray, x_test)
    )
    
    # Save the model
    model.save('colorization_model.h5')
    
    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    return model

# Part 2: Colorize a New Image
def colorize_image(model, image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.reshape(1, 32, 32, 1)
    
    # Predict colorized image
    colorized = model.predict(img_gray)
    colorized = colorized[0] * 255.0  # Denormalize
    colorized = colorized.astype('uint8')
    
    # Save and display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Grayscale Input')
    plt.imshow(img_gray[0, :, :, 0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Colorized Output')
    plt.imshow(colorized)
    plt.axis('off')
    
    plt.savefig('colorization_result.png')
    plt.close()
    
    return colorized

if __name__ == '__main__':
    # Train the model
    model = train_model()
    
    # Test colorization (replace 'test_image.jpg' with your image path)
    # Ensure the image is a 32x32 RGB image for compatibility
    if os.path.exists('test_image.jpg'):
        colorize_image(model, 'test_image.jpg')