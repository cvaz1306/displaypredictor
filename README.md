# DisplayPred: Face Landmark Detection and Prediction

DisplayPred is a project that involves training a neural network model to predict the position of the mouse cursor based on facial landmarks detected in real-time. The project consists of three main components: training the model (`trainer.py`), making predictions in real-time (`inference.py`), and collecting data for training (`collection.py`).

## Table of Contents

- [Trainer.py](#trainerpy)
- [Inference.py](#inferencepy)
- [Collection.py](#collectionpy)
- [Requirements](#requirements)
- [Usage](#usage)

## Trainer.py

`trainer.py` is responsible for training the neural network model. It uses a dataset of facial landmarks to train a complex model that can predict the position of the mouse cursor based on the detected landmarks.

### Usage

1. Prepare your dataset in CSV format with facial landmarks.
2. Run `trainer.py` with the following command-line arguments:
   - `-i` or `--input`: Path to the dataset CSV file.
   - `-o` or `--output`: Path to save the trained model.
   - `-e` or `--epochs`: Number of epochs for training.
   - `-lr` or `--learning-rate`: Learning rate for the optimizer.

Example:

```bash
python trainer.py -i datasets/output.csv -o models/model.pth -e 200000 -lr 0.000001
```

## Inference.py

`inference.py` uses the trained model to make real-time predictions about the position of the mouse cursor based on facial landmarks detected in the webcam feed.

### Usage

1. Ensure you have a trained model saved from `trainer.py`.
2. Run `inference.py` with the following command-line arguments:
   - `-m` or `--model`: Path to the trained model.
   - `-b` or `--bias`: Bias value to adjust the prediction.

Example:

```bash
python inference.py -m models/model.pth -b -.1
```

## Collection.py

`collection.py` is a utility script for collecting facial landmark data for training the model. It displays a window on each monitor with an eye icon, and the user can click the spacebar to capture the current facial landmarks.

### Usage

1. Run `collection.py` with the following command-line argument:
   - `-o` or `--output`: Path to save the collected data.

Example:

```bash
python collection.py -o datasets/output.csv
```

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- Dlib
- Tkinter
- Pillow
- screeninfo
- pyautogui

## Usage

1. Collect data using `collection.py`.
2. Train the model using `trainer.py`.
3. Use `inference.py` for real-time predictions.

---

This README provides a high-level overview of each component. For more detailed instructions and explanations, please refer to the comments and documentation within each file.
