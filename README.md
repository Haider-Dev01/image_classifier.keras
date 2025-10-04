# CIFAR-10 Image Classifier

This project is a simple image classifier for the CIFAR-10 dataset using TensorFlow/Keras and OpenCV. It demonstrates how to train a convolutional neural network (CNN) on CIFAR-10, save the trained model, and use it to predict the class of a custom image.

## Folder Structure

```
car.jpg
cifar10_classifier.keras
deer.jpg
horse.jpg
main.py
plane.jpg
training.py
```

- `main.py`: Loads the trained model and predicts the class of a given image (e.g., `horse.jpg`).
- `training.py`: Trains a CNN on the CIFAR-10 dataset and saves the model.
- `*.jpg`: Example images for prediction.
- `cifar10_classifier.keras`: The saved Keras model.

## How to Use

### 1. Train the Model

Run `training.py` to train the model on a subset of CIFAR-10 and save it:

```sh
python training.py
```

- Loads CIFAR-10 dataset from Keras.
- Normalizes the images to [0, 1].
- Shows 16 sample images with their class names.
- Trains a CNN on 10,000 training images and 2,000 test images.
- Evaluates and saves the model as `cifar10_classifier.keras`.

### 2. Predict an Image

Run `main.py` to load the saved model and predict the class of an image:

```sh
python main.py
```

- Loads the trained model.
- Reads an image (e.g., `horse.jpg`) using OpenCV.
- Converts the image from BGR to RGB.
- Displays the image in grayscale.
- Predicts the class using the model and prints the predicted class name.

**Note:** The code assumes the input image is 32x32 pixels with 3 color channels (RGB), matching CIFAR-10's format. If your image is a different size, you may need to resize it before prediction.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (`cv2`)
- NumPy
- Matplotlib

Install dependencies with:

```sh
pip install tensorflow opencv-python numpy matplotlib
```

## Class Names

The CIFAR-10 classes used are:

- plane
- car
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Comments Verification

- The comments in the code accurately describe each step, including data loading, normalization, visualization, model creation, training, evaluation, saving, and prediction.
- The image is converted from BGR to RGB before prediction, and displayed in grayscale for visualization.
- The model expects images normalized to [0, 1].

---

Feel free to use or modify this project for your own experiments!
