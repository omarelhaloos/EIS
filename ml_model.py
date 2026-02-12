"""
ML Model Module for EIS Regression
Builds and evaluates a 1D-CNN regression model for predicting EIS circuit parameters.
Extracted from the regression model Jupyter notebook.

Author: Dulyawat Doonyapisut (charting9@gmail.com)
"""

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# TensorFlow is optional — may not be available on all Python versions
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None


# Parameter names for Circuit 4 regression output
PARAM_NAMES = ["Rs", "R1", "R2", "Q1", "Q2", "Sigma"]


def load_and_preprocess_data(file_path_or_buffer, test_size=0.2, random_state=42, is_test=False):
    """
    Load .mat file and preprocess EIS data for the regression model.
    
    Args:
        file_path_or_buffer: path to .mat file or file-like object
        test_size: fraction for test split
        random_state: random seed
        is_test: if True, apply test-set scaling (10^6 for Q columns)
        
    Returns:
        x_train, x_test, y_train, y_test
    """
    mat = scipy.io.loadmat(file_path_or_buffer)
    x = mat["x_data"]
    y = mat["y_data"]
    y = np.squeeze(y)
    x = np.swapaxes(x, 1, 2)

    # Augmentation: add negated channels
    new_shape = np.asarray(x.shape)
    new_shape[-1] = new_shape[-1] + 3
    new_shape = tuple(new_shape)
    new_x = np.zeros(new_shape)
    new_x[:, :, :3] = x

    # Scale Q columns for numerical stability
    if is_test:
        y[:, 4] = y[:, 4] * 10**6
        y[:, 6] = y[:, 6] * 10**6
    else:
        y[:, 3] = y[:, 3] * 10**4
        y[:, 4] = y[:, 4] * 10**7
        y[:, 5] = y[:, 5] * 10**4
        y[:, 6] = y[:, 6] * 10**7

    y = np.delete(y, [3, 5], axis=1)

    # Augmented channels (negated)
    new_x[:, :, 3] = x[:, :, 0] * -1
    new_x[:, :, 4] = x[:, :, 1] * -1
    new_x[:, :, 5] = x[:, :, 2] * -1

    x_train, x_test, y_train, y_test = train_test_split(
        new_x, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def make_model(input_shape):
    """
    Build the Conv1D → Dense regression model.
    
    Architecture:
        5x Conv1D layers (64→128→256→512→768 filters)
        Dense layers (512→512→BN→Flatten→64→64)
        Output: 6 parameters
    """
    initializer = tf.keras.initializers.HeNormal()

    input_layer = keras.layers.Input(input_shape)

    conv1d = keras.layers.Conv1D(
        filters=64, kernel_size=32, padding="same", activation="relu",
        kernel_initializer=initializer
    )(input_layer)

    conv1d = keras.layers.Conv1D(
        filters=128, kernel_size=16, padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    conv1d = keras.layers.Conv1D(
        filters=256, kernel_size=8, padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    conv1d = keras.layers.Conv1D(
        filters=512, kernel_size=4, padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    conv1d = keras.layers.Conv1D(
        filters=768, kernel_size=2, padding="same", activation="relu",
        kernel_initializer=initializer
    )(conv1d)

    connector = conv1d

    dense1 = keras.layers.Dense(
        512, activation="linear", kernel_initializer=initializer
    )(connector)

    dense1 = keras.layers.Dense(
        512, activation="linear", kernel_initializer=initializer
    )(dense1)

    dense1 = keras.layers.BatchNormalization()(dense1)
    dense1 = keras.layers.Flatten()(dense1)

    dense1 = keras.layers.Dense(
        64, activation="linear", kernel_initializer=initializer
    )(dense1)

    dense1 = keras.layers.Dense(
        64, activation="linear", kernel_initializer=initializer
    )(dense1)

    output_layer = keras.layers.Dense(6)(dense1)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def evaluate_model(model, x_test, y_test, n_samples=100):
    """
    Evaluate model predictions and compute per-parameter metrics.
    
    Returns:
        y_pred: predicted values
        metrics: dict of {param_name: {r2, mae, mape, mse}}
    """
    y_pred = model.predict(x_test)
    y_pred = np.asarray(y_pred)

    metrics = {}
    for i, name in enumerate(PARAM_NAMES):
        a = y_test[:n_samples, i]
        b = y_pred[:n_samples, i]
        metrics[name] = {
            "R²": r2_score(a, b),
            "MAE": mean_absolute_error(a, b),
            "MAPE (%)": mean_absolute_percentage_error(a, b) * 100,
            "MSE": mean_squared_error(a, b),
        }

    return y_pred, metrics
