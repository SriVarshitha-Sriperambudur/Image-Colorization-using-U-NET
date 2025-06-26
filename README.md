# Image-Colorization-using-U-NET

This project implements an image colorization system using a U-Net architecture trained on the CIFAR-10 dataset. The model learns to convert grayscale images into colorized RGB versions, capturing contextual and spatial features via a deep convolutional encoder-decoder network.

🧠 Key Features:

🧾 Trains on CIFAR-10 dataset (32x32 RGB images)

🔳 Converts grayscale input images to colorized outputs

🌈 Uses a U-Net model with skip connections for accurate reconstruction

📊 Plots training vs. validation loss after training

🖼️ Colorizes new grayscale images using the trained model

💾 Saves output visualizations and model files

🧰 Technologies Used:

Python 3.x

TensorFlow / Keras

OpenCV

Matplotlib

NumPy

CIFAR-10 dataset

🧠 Model Architecture: U-Net

Encoder: 2 downsampling stages with Conv2D + MaxPooling2D

Bottleneck: Deep representation with 256 filters

Decoder: 2 upsampling stages using UpSampling2D + Concatenate

Output: 3-channel RGB image (32x32x3) using sigmoid activation

