"""
Corrosion Predictor Module
Industrial-grade corrosion rate prediction from EIS spectrum data
and environmental conditions.

No TensorFlow dependency — uses joblib for sklearn model loading.
"""

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "Low": {"max": 0.1, "color": "#10b981", "bg": "rgba(16,185,129,0.12)", "border": "rgba(16,185,129,0.3)"},
    "Moderate": {"max": 0.5, "color": "#f59e0b", "bg": "rgba(245,158,11,0.12)", "border": "rgba(245,158,11,0.3)"},
    "Severe": {"max": float("inf"), "color": "#ef4444", "bg": "rgba(239,68,68,0.12)", "border": "rgba(239,68,68,0.3)"},
}


def classify_risk(corrosion_rate: float) -> tuple:
    """
    Classify corrosion rate into risk level.

    Returns:
        (risk_label, color_hex, bg_rgba, border_rgba)
    """
    if corrosion_rate < 0.1:
        r = RISK_THRESHOLDS["Low"]
        return "Low", r["color"], r["bg"], r["border"]
    elif corrosion_rate < 0.5:
        r = RISK_THRESHOLDS["Moderate"]
        return "Moderate", r["color"], r["bg"], r["border"]
    else:
        r = RISK_THRESHOLDS["Severe"]
        return "Severe", r["color"], r["bg"], r["border"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(file_buffer):
    """
    Load a sklearn model from a .pkl file buffer.

    Returns the model object.
    Raises ValueError if the file is not a valid sklearn model.
    """
    try:
        model = joblib.load(file_buffer)
    except Exception as e:
        raise ValueError(f"Failed to load model file: {e}")

    if not hasattr(model, "predict"):
        raise ValueError(
            "The uploaded file does not contain a valid sklearn model "
            "(missing .predict() method)."
        )
    return model


# ---------------------------------------------------------------------------
# Spectrum loading
# ---------------------------------------------------------------------------
def load_spectrum(file_buffer) -> np.ndarray:
    """
    Load an EIS spectrum from a CSV file.

    Returns a 2D numpy array (rows × columns).
    Raises ValueError on invalid format.
    """
    try:
        df = pd.read_csv(file_buffer)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Convert to numeric, coercing errors
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError(
            "The CSV file does not contain any numeric columns. "
            "Please upload a file with numeric EIS spectrum data."
        )

    return numeric_df.values


def load_mat_spectrum(file_buffer) -> np.ndarray:
    """
    Load an EIS spectrum from a .mat file and apply the SAME preprocessing
    used during model training (load_and_preprocess_data in ml_model.py):

        1. Extract x_data  → shape (N, channels, freq_points)
        2. swapaxes(1, 2)  → shape (N, freq_points, channels)
        3. Augment: append negated channels → (N, freq_points, 2*channels)
        4. Take the FIRST sample and flatten → (1, freq_points * 2 * channels)

    This produces the exact feature count the GradientBoostingRegressor expects
    (e.g. 100 freq_points × 3 channels × 2 = 600 features).
    """
    import scipy.io

    try:
        mat = scipy.io.loadmat(file_buffer)
    except Exception as e:
        raise ValueError(f"Failed to read .mat file: {e}")

    if "x_data" not in mat:
        raise ValueError(
            "The .mat file must contain an 'x_data' variable "
            "(same format used for model training)."
        )

    x = mat["x_data"]  # (N, channels, freq_points)

    # Replicate training preprocessing
    x = np.swapaxes(x, 1, 2)  # → (N, freq_points, channels)

    n_channels = x.shape[-1]
    # Augment with negated channels (matches training pipeline)
    augmented = np.zeros((*x.shape[:-1], n_channels * 2))
    augmented[:, :, :n_channels] = x
    for ch in range(n_channels):
        augmented[:, :, n_channels + ch] = x[:, :, ch] * -1

    # Take first sample, flatten → (1, freq_pts * 2 * channels)
    sample = augmented[0].flatten().astype(np.float64)
    return sample.reshape(1, -1)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

# Environmental feature names (must match training order)
ENV_FEATURES = [
    "temperature_c", "pressure_bar", "ph",
    "sulfur_ppm", "flow_velocity_ms", "service_years",
]

# Material encoding map (one-hot style matching typical sklearn pipelines)
MATERIAL_TYPES = [
    "Alloy Steel", "Carbon Steel", "Inconel", "SS304", "SS316",
]


def encode_material(material: str) -> list:
    """One-hot encode material type."""
    return [1.0 if m == material else 0.0 for m in MATERIAL_TYPES]


def build_feature_vector(
    spectrum: np.ndarray,
    material: str,
    temperature: float,
    pressure: float,
    ph: float,
    sulfur: float,
    flow_velocity: float,
    service_years: int,
) -> dict:
    """
    Build candidate feature vectors for prediction.

    Returns a dict with three candidate arrays so that predict_corrosion
    can pick the one matching the model's expected feature count:
        - "full"     : flattened spectrum + material one-hot + env params
        - "spectrum" : flattened spectrum only
        - "env"      : material one-hot + env params only

    Each value is a 2D array of shape (1, n_features).
    """
    flat_spectrum = spectrum.flatten().astype(np.float64)

    material_encoded = encode_material(material)
    env_features = [
        temperature, pressure, ph, sulfur, flow_velocity, float(service_years),
    ]
    env_arr = np.array(material_encoded + env_features, dtype=np.float64)

    full = np.concatenate([flat_spectrum, env_arr])
    return {
        "full": full.reshape(1, -1),
        "spectrum": flat_spectrum.reshape(1, -1),
        "env": env_arr.reshape(1, -1),
    }


def predict_corrosion(model, features: dict) -> float:
    """
    Run prediction, automatically selecting the feature vector that
    matches the model's expected input size.

    Parameters:
        model    : a fitted sklearn estimator with .predict()
        features : dict returned by build_feature_vector()

    Returns the predicted corrosion rate as a float.
    """
    # Determine expected feature count from the model
    expected = getattr(model, "n_features_in_", None)

    # Try candidates in preference order
    for key in ("full", "spectrum", "env"):
        vec = features[key]
        n = vec.shape[1]
        if expected is not None and n == expected:
            prediction = model.predict(vec)
            return float(np.asarray(prediction).flatten()[0])

    # If no exact match found, try full vector anyway (let sklearn raise
    # a clear error with actual vs expected counts)
    try:
        prediction = model.predict(features["full"])
        return float(np.asarray(prediction).flatten()[0])
    except ValueError:
        # Last resort: try spectrum-only
        prediction = model.predict(features["spectrum"])
        return float(np.asarray(prediction).flatten()[0])


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------
def create_gauge_chart(corrosion_rate: float, risk_label: str, risk_color: str) -> go.Figure:
    """
    Create a Plotly gauge chart showing the predicted corrosion rate.
    """
    # Dynamic max for gauge scale
    gauge_max = max(1.5, corrosion_rate * 1.5)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=corrosion_rate,
        number=dict(
            font=dict(size=48, color="#e8edf5", family="Inter, sans-serif"),
            suffix=" mm/yr",
        ),
        title=dict(
            text=f"<b>Corrosion Rate</b><br><span style='font-size:0.85em; color:{risk_color}'>"
                 f"Risk: {risk_label}</span>",
            font=dict(size=18, color="#e8edf5"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, gauge_max],
                tickwidth=2,
                tickcolor="#4b5563",
                dtick=round(gauge_max / 5, 2),
            ),
            bar=dict(color=risk_color, thickness=0.7),
            bgcolor="rgba(13,17,23,0.6)",
            borderwidth=2,
            bordercolor="rgba(99,102,241,0.2)",
            steps=[
                dict(range=[0, 0.1], color="rgba(16,185,129,0.15)"),
                dict(range=[0.1, 0.5], color="rgba(245,158,11,0.15)"),
                dict(range=[0.5, gauge_max], color="rgba(239,68,68,0.15)"),
            ],
            threshold=dict(
                line=dict(color="#e8edf5", width=3),
                thickness=0.8,
                value=corrosion_rate,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e8edf5"),
        height=350,
        margin=dict(l=30, r=30, t=80, b=30),
    )

    return fig


def get_env_ranges(csv_path: str) -> dict:
    """
    Read the corrosion pipeline dataset and return min/max/unique values
    for environmental condition selectors.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Fallback defaults if file is missing
        return {
            "materials": MATERIAL_TYPES,
            "temperature_c": (140.0, 280.0),
            "pressure_bar": (25.0, 55.0),
            "ph": (4.0, 7.0),
            "sulfur_ppm": (100, 700),
            "flow_velocity_ms": (1.5, 4.5),
            "service_years": (1, 15),
        }

    return {
        "materials": sorted(df["material"].unique().tolist()),
        "temperature_c": (float(df["temperature_c"].min()), float(df["temperature_c"].max())),
        "pressure_bar": (float(df["pressure_bar"].min()), float(df["pressure_bar"].max())),
        "ph": (float(df["ph"].min()), float(df["ph"].max())),
        "sulfur_ppm": (int(df["sulfur_ppm"].min()), int(df["sulfur_ppm"].max())),
        "flow_velocity_ms": (float(df["flow_velocity_ms"].min()), float(df["flow_velocity_ms"].max())),
        "service_years": (int(df["service_years"].min()), int(df["service_years"].max())),
    }
