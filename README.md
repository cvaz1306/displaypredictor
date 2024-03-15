# DisplayPred: Face Landmark Detection and Prediction
## Installation and Setup

This section provides instructions on how to set up your development environment for the DisplayPred project. It includes creating a virtual environment and installing all necessary Python libraries.

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

First, clone the DisplayPredictor repository to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/cvaz1306/displaypredictor.git
```

### Step 2: Create a Virtual Environment

Navigate to the project directory:

```bash
cd displaypredictor
```

Create a virtual environment named `venv` (or any name you prefer):

```bash
python3 -m venv venv
```

Activate the virtual environment:

- On Windows:

```bash
.\venv\Scripts\activate
```

- On macOS and Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

With the virtual environment activated, install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This command installs all the necessary libraries listed in the `requirements.txt` file, including PyTorch, OpenCV, Dlib, Tkinter, Pillow, screeninfo, and pyautogui.

### Step 4: Verify Installation

To verify that all libraries have been installed correctly, you can run a simple Python script that imports each library. Run `test_imports` in your project directory:


```bash
python test_imports.py
```

If the script runs without any import errors, your installation is successful.

### Step 5: Running the Project

Now that your environment is set up, you can run the DisplayPredictor scripts (`trainer.py`, `inference.py`, and `collection.py`) as described in the project documentation.

---

This setup ensures that your project's dependencies are isolated from other Python projects, making it easier to manage and update them without affecting other projects.

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
