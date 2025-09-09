# Handwriting Digit Recognizer

This project is a web application that uses a deep learning model to recognize handwritten digits. You can draw a digit on a canvas, and the application will instantly predict what number it is with a confidence score.

The application is built with **Marimo**, **Streamlit** and the deep learning model is powered by **TensorFlow/Keras**.

-----

## 🚀 How it Works

The project consists of two main parts:

1.  **A Convolutional Neural Network (CNN) Model**: The model is trained to classify images of handwritten digits. It was trained on the **MNIST** dataset, a classic dataset in computer vision.
2.  **A Streamlit Web Application**: This app provides a simple and intuitive user interface where you can draw a digit. The app preprocesses the image you draw and feeds it to the trained model to get a prediction.

-----

## 🛠️ Project Structure

The project directory is organized as follows:

```
Handwritten-Digit-Recognizer/
├── mnist_cnn_model.keras   # The trained Keras model
├── streamlit_app.py        # The Streamlit web application
├── requirements.txt        # Python dependencies for the project
├── README.md               # This README file
└── mnist_model_notebook.ipynb  # Jupyter notebook for model training and evaluation
```

  - **`mnist_cnn_model.keras`**: This file contains the architecture, weights, and training configuration of the trained CNN model.
  - **`streamlit_app.py`**: This is the main application file. It loads the model, sets up the canvas, and handles the prediction logic.
  - **`requirements.txt`**: This file lists all the Python libraries required to run the application.

-----

## 📄 Getting Started

### Prerequisites

Make sure you have Python installed on your system. It's highly recommended to use a virtual environment.

### Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/issa-kabore/Handwritten-Digit-Recognizer.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Handwritten-Digit-Recognizer
    ```
3.  Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Once the dependencies are installed, you can launch the Streamlit app from your terminal:

```bash
streamlit run streamlit_app.py
```

This will start a local web server and open the application in your default web browser.

-----

## 📊 Model Training and Evaluation

The deep learning model was developed and trained in a **marimo** notebook, which was then exported to the `mnist_model_notebook.ipynb` file for easy viewing on GitHub. The notebook covers the following steps:

  - **Data Loading and Preprocessing**: Normalizing and reshaping the MNIST images.
  - **Model Architecture**: Defining a CNN with `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
  - **Training**: Using techniques like **Data Augmentation** and **Early Stopping** to prevent overfitting.
  - **Evaluation**: Analyzing the model's performance with a **Confusion Matrix** and a **Classification Report**.

### A propos de Marimo

The notebook code was initially written using **marimo**, a modern framework for Python notebooks. Unlike traditional notebooks such as Jupyter, marimo uses a **reactive** model where code cells that depend on other cells are automatically updated when the source data changes. This approach ensures code **reproducibility** and a consistent notebook state.

-----

## 🔗 Links

  * **Live App**: [handwritten-digit-recognizer](https://handwritten-digit-recognizer-qbgwuqu3xzahjufiq7zyvy.streamlit.app/)
  * **Source Notebook (marimo)**: [static.marimo.app/static/mnist-digit-recognizer](https://static.marimo.app/static/mnist-digit-recognizer-gkmn)

-----

## ☁️ Deployment

This application is deployed live on [**Streamlit Cloud**](https://streamlit.io/cloud). To deploy your own version, simply push your code to a public GitHub repository and connect it to your [Streamlit Cloud account](https://share.streamlit.io/).