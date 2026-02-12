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

    # --- FIX: ensure y is always 2D (samples × params) ---
    # np.squeeze can collapse to 1D when there is only 1 sample or 1 param.
    # Instead, remove only truly extraneous leading/trailing dims and guarantee 2D.
    y = np.atleast_2d(np.squeeze(y))
    # If squeeze + atleast_2d produced (params, 1) instead of (1, params) for a
    # single-sample file, transpose so rows = samples.
    if y.shape[0] == 1 and y.ndim == 2:
        pass  # already (1, params)
    elif y.ndim == 2 and y.shape[1] == 1 and y.shape[0] > 1:
        y = y.T  # was stored column-wise for a single sample

    x = np.swapaxes(x, 1, 2)

    n_input_channels = x.shape[-1]

    # Augmentation: add negated channels
    new_shape = list(x.shape)
    new_shape[-1] = n_input_channels * 2
    new_x = np.zeros(new_shape)
    new_x[:, :, :n_input_channels] = x

    n_ycols = y.shape[1]

    # Scale Q columns for numerical stability
    # The original notebook expects 8 columns for the non-test path (Circuit 4 full)
    # and 6 or 8 columns for the test path, but .mat files from the simulator
    # may already have the reduced 6-column layout.
    if n_ycols >= 7:
        # Full 8-column layout: cols 3,5 = alpha; cols 4,6 = Q
        if is_test:
            y[:, 4] = y[:, 4] * 10**6
            y[:, 6] = y[:, 6] * 10**6
        else:
            y[:, 3] = y[:, 3] * 10**4
            y[:, 4] = y[:, 4] * 10**7
            y[:, 5] = y[:, 5] * 10**4
            y[:, 6] = y[:, 6] * 10**7
        # Remove alpha columns (indices 3 and 5)
        y = np.delete(y, [3, 5], axis=1)
    # else: y already has ≤6 columns (pre-processed or different circuit), use as-is

    # Augmented channels (negated)
    for ch in range(n_input_channels):
        new_x[:, :, n_input_channels + ch] = x[:, :, ch] * -1

    x_train, x_test, y_train, y_test = train_test_split(
        new_x, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def make_model(input_shape, n_outputs=6):
    """
    Build an optimized Conv1D → Dense regression model.

    Architecture (with regularisation):
        5× Conv1D layers (64→128→256→512→768 filters) with BatchNorm + Dropout
        Dense layers (512→512→BN→Flatten→128→64) with Dropout
        Output: n_outputs parameters

    Args:
        input_shape: shape of a single input sample, e.g. (100, 6)
        n_outputs: number of regression targets (default 6)
    """
    initializer = tf.keras.initializers.HeNormal()

    input_layer = keras.layers.Input(input_shape)

    # --- Conv blocks ---
    conv_configs = [
        (64,  32),
        (128, 16),
        (256,  8),
        (512,  4),
        (768,  2),
    ]

    x = input_layer
    for filters, kernel_size in conv_configs:
        x = keras.layers.Conv1D(
            filters=filters, kernel_size=kernel_size, padding="same",
            activation="relu", kernel_initializer=initializer
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)

    # --- Dense head ---
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=initializer)(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(128, activation="relu", kernel_initializer=initializer)(x)
    x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Dense(64, activation="relu", kernel_initializer=initializer)(x)

    output_layer = keras.layers.Dense(n_outputs)(x)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def get_training_callbacks(patience_lr=8, patience_es=20, min_lr=1e-6):
    """
    Return a list of Keras callbacks for training:
      - ReduceLROnPlateau: halve LR when val_loss stalls
      - EarlyStopping: stop training if no improvement
    """
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=patience_lr,
            min_lr=min_lr, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es,
            restore_best_weights=True, verbose=1,
        ),
    ]
    return callbacks


def evaluate_model(model, x_test, y_test, n_samples=100):
    """
    Evaluate model predictions and compute per-parameter metrics.

    Returns:
        y_pred: predicted values
        metrics: dict of {param_name: {r2, mae, mape, mse}}
    """
    y_pred = model.predict(x_test)
    y_pred = np.asarray(y_pred)

    # Determine how many parameters to evaluate
    n_params = min(y_test.shape[1], len(PARAM_NAMES))

    metrics = {}
    for i in range(n_params):
        name = PARAM_NAMES[i]
        a = y_test[:n_samples, i]
        b = y_pred[:n_samples, i]
        metrics[name] = {
            "R²": r2_score(a, b),
            "MAE": mean_absolute_error(a, b),
            "MAPE (%)": mean_absolute_percentage_error(a, b) * 100,
            "MSE": mean_squared_error(a, b),
        }

    return y_pred, metrics
