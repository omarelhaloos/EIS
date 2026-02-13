"""
EIS Data Simulation Module
Simulates Electrochemical Impedance Spectroscopy data for various circuit models.
Extracted from the EIS data simulation Jupyter notebook.

Author: Dulyawat Doonyapisut (charting9@gmail.com)
"""

import numpy as np
import scipy.io

# =============================================================================
# Essential Elements
# =============================================================================

def F_range(initial_frequency, last_frequency, number_of_point=100):
    """
    Define frequency range in log scale.
    
    Returns:
        angular_frequency: log scale angular frequency [s^-1]
        frequency_Hz: log scale frequency [Hz]
    """
    frequency_Hz = np.logspace(
        np.log10(initial_frequency),
        np.log10(last_frequency),
        number_of_point,
        endpoint=True,
    )
    angular_frequency = 2 * np.pi * frequency_Hz
    return angular_frequency, frequency_Hz


def Z_R(resistance):
    """Impedance of a resistor [Ohm]."""
    return resistance


def Z_Q(non_ideal_capacitance, ideality_factor, angular_frequency):
    """
    Impedance of a Constant Phase Element (CPE) [Ohm].
    Z_Q = 1 / (Q * (jω)^α)
    """
    return 1 / (non_ideal_capacitance * (angular_frequency * 1j) ** ideality_factor)


def Z_W(sigma, angular_frequency):
    """
    Impedance of an Infinite Warburg Element [Ohm].
    Z_W = σ√2 / √(jω)
    """
    return (sigma * np.sqrt(2)) / np.sqrt(1j * angular_frequency)


# =============================================================================
# Random generators
# =============================================================================

def log_rand(initial_gen, last_gen, size_number):
    """Random values in log space."""
    initial_v = np.log(initial_gen)
    last_v = np.log(last_gen)
    return np.exp(initial_v + (last_v - initial_v) * np.random.rand(size_number))


def lin_rand(initial_gen, last_gen, size_number):
    """Random values in linear space."""
    return initial_gen + (last_gen - initial_gen) * np.random.rand(size_number)


# =============================================================================
# Array generators
# =============================================================================

def genZR(size_number, number_of_point, resistance):
    """Generate resistance impedance array."""
    ZR = np.zeros((size_number, number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point):
            ZR[idx][idx2] = Z_R(resistance[idx])
    return ZR


def genZQ(size_number, number_of_point, non_ideal_capacitance, ideality_factor, angular_frequency):
    """Generate CPE impedance array."""
    ZQ = np.zeros((size_number, number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point):
            ZQ[idx][idx2] = Z_Q(non_ideal_capacitance[idx], ideality_factor[idx], angular_frequency[idx2])
    return ZQ


def genZW(size_number, number_of_point, sigma, angular_frequency):
    """Generate Warburg impedance array."""
    ZW = np.zeros((size_number, number_of_point), dtype=complex)
    for idx in range(size_number):
        for idx2 in range(number_of_point):
            ZW[idx][idx2] = Z_W(sigma[idx], angular_frequency[idx2])
    return ZW


# =============================================================================
# Circuit Simulations
# =============================================================================

# Circuit descriptions for the UI
CIRCUIT_INFO = {
    1: {
        "name": "Circuit 1: R₁ + (R₂ ∥ Q₁)",
        "description": "Series resistor with one R//CPE loop",
        "params": ["R1", "R2", "α₁", "Q1"],
    },
    2: {
        "name": "Circuit 2: R₁ + (R₂ ∥ Q₁) + (R₃ ∥ Q₂)",
        "description": "Series resistor with two R//CPE loops",
        "params": ["R1", "R2", "R3", "α₁", "Q1", "α₂", "Q2"],
    },
    3: {
        "name": "Circuit 3: R₁ + (Q₁ ∥ (R₂ + W))",
        "description": "Series resistor with CPE parallel to R+Warburg",
        "params": ["R1", "R2", "α₁", "Q1", "σ"],
    },
    4: {
        "name": "Circuit 4: R₁ + (R₂ ∥ Q₁) + (Q₂ ∥ (R₃ + W))",
        "description": "Two loops: one R//CPE and one CPE//(R+Warburg)",
        "params": ["R1", "R2", "R3", "α₁", "Q1", "α₂", "Q2", "σ"],
    },
    5: {
        "name": "Circuit 5: R₁ + ((R₂ + ((R₃ + W) ∥ Q₂)) ∥ Q₁)",
        "description": "Nested circuit with Warburg diffusion",
        "params": ["R1", "R2", "R3", "α₁", "Q1", "α₂", "Q2", "σ"],
    },
}


def sim_circuit(
    circuit_id,
    size_number,
    number_of_point,
    angular_frequency,
    resistance_range,
    alpha_range,
    q_range,
    sigma_range,
):
    """
    Simulate a specific circuit.
    
    Returns:
        Zsum: complex impedance array (size_number, number_of_point)
        Zparam: parameter array
    """
    if circuit_id == 1:
        return _sim_cir1(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range)
    elif circuit_id == 2:
        return _sim_cir2(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range)
    elif circuit_id == 3:
        return _sim_cir3(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range, sigma_range)
    elif circuit_id == 4:
        return _sim_cir4(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range, sigma_range)
    elif circuit_id == 5:
        return _sim_cir5(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range, sigma_range)
    else:
        raise ValueError(f"Unknown circuit_id: {circuit_id}")


def _sim_cir1(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range):
    R1 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr1 = genZR(size_number, number_of_point, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr2 = genZR(size_number, number_of_point, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q1 = log_rand(q_range[0], q_range[1], size_number)
    Zq1 = genZQ(size_number, number_of_point, Q1, ideality_factor1, angular_frequency)

    Zsum = Zr1 + 1 / (1 / Zr2 + 1 / Zq1)

    Zparam = []
    for idx in range(size_number):
        Zparam.append([R1[idx], R2[idx], ideality_factor1[idx], Q1[idx]])

    return Zsum, np.array(Zparam)


def _sim_cir2(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range):
    R1 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr1 = genZR(size_number, number_of_point, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr2 = genZR(size_number, number_of_point, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q1 = log_rand(q_range[0], q_range[1], size_number)
    Zq1 = genZQ(size_number, number_of_point, Q1, ideality_factor1, angular_frequency)

    R3 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr3 = genZR(size_number, number_of_point, R3)

    ideality_factor2 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q2 = log_rand(q_range[0], q_range[1], size_number)
    Zq2 = genZQ(size_number, number_of_point, Q2, ideality_factor1, angular_frequency)

    Zsum = Zr1 + 1 / (1 / Zr2 + 1 / Zq1) + 1 / (1 / Zr3 + 1 / Zq2)

    Zparam = []
    for idx in range(size_number):
        Zparam.append([R1[idx], R2[idx], R3[idx], ideality_factor1[idx], Q1[idx], ideality_factor2[idx], Q2[idx]])

    return Zsum, np.array(Zparam)


def _sim_cir3(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range, sigma_range):
    R1 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr1 = genZR(size_number, number_of_point, R1)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q1 = log_rand(q_range[0], q_range[1], size_number)
    Zq1 = genZQ(size_number, number_of_point, Q1, ideality_factor1, angular_frequency)

    R2 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr2 = genZR(size_number, number_of_point, R2)

    sigma = log_rand(sigma_range[0], sigma_range[1], size_number)
    Zw = genZW(size_number, number_of_point, sigma, angular_frequency)

    Zsum = Zr1 + 1 / (1 / Zq1 + 1 / (Zr2 + Zw))

    Zparam = []
    for idx in range(size_number):
        Zparam.append([R1[idx], R2[idx], ideality_factor1[idx], Q1[idx], sigma[idx]])

    return Zsum, np.array(Zparam)


def _sim_cir4(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range, sigma_range):
    R1 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr1 = genZR(size_number, number_of_point, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr2 = genZR(size_number, number_of_point, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q1 = log_rand(q_range[0], q_range[1], size_number)
    Zq1 = genZQ(size_number, number_of_point, Q1, ideality_factor1, angular_frequency)

    ideality_factor2 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q2 = log_rand(q_range[0], q_range[1], size_number)
    Zq2 = genZQ(size_number, number_of_point, Q2, ideality_factor1, angular_frequency)

    R3 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr3 = genZR(size_number, number_of_point, R3)

    sigma = log_rand(sigma_range[0], sigma_range[1], size_number)
    Zw = genZW(size_number, number_of_point, sigma, angular_frequency)

    Zsum = Zr1 + 1 / (1 / Zr2 + 1 / Zq1) + 1 / (1 / Zq2 + 1 / (Zr3 + Zw))

    Zparam = []
    for idx in range(size_number):
        Zparam.append([R1[idx], R2[idx], R3[idx], ideality_factor1[idx], Q1[idx], ideality_factor2[idx], Q2[idx], sigma[idx]])

    return Zsum, np.array(Zparam)


def _sim_cir5(size_number, number_of_point, angular_frequency, resistance_range, alpha_range, q_range, sigma_range):
    R1 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr1 = genZR(size_number, number_of_point, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr2 = genZR(size_number, number_of_point, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q1 = log_rand(q_range[0], q_range[1], size_number)
    Zq1 = genZQ(size_number, number_of_point, Q1, ideality_factor1, angular_frequency)

    R3 = log_rand(resistance_range[0], resistance_range[1], size_number)
    Zr3 = genZR(size_number, number_of_point, R3)

    ideality_factor2 = np.round(lin_rand(alpha_range[0], alpha_range[1], size_number), 3)
    Q2 = log_rand(q_range[0], q_range[1], size_number)
    Zq2 = genZQ(size_number, number_of_point, Q2, ideality_factor1, angular_frequency)

    sigma = log_rand(sigma_range[0], sigma_range[1], size_number)
    Zw = genZW(size_number, number_of_point, sigma, angular_frequency)

    Zsum = Zr1 + 1 / (1 / (Zr2 + 1 / ((1 / (Zr3 + Zw)) + 1 / Zq2)) + 1 / Zq1)

    Zparam = []
    for idx in range(size_number):
        Zparam.append([R1[idx], R2[idx], R3[idx], ideality_factor1[idx], Q1[idx], ideality_factor2[idx], Q2[idx], sigma[idx]])

    return Zsum, np.array(Zparam)


# =============================================================================
# Data export helpers
# =============================================================================

def arrange_data(Circuit, cir_class, size_number, number_of_point):
    """Arrange a single circuit's data into feature arrays."""
    imge = Circuit.imag
    phase = np.degrees(np.arctan(Circuit.imag / Circuit.real))
    mag = np.absolute(Circuit)

    x = np.zeros((size_number, 3, number_of_point))
    y = np.zeros(size_number)

    for idx in range(size_number):
        y[idx] = cir_class
        for idx2 in range(number_of_point):
            x[idx][0][idx2] = imge[idx][idx2]
            x[idx][1][idx2] = phase[idx][idx2]
            x[idx][2][idx2] = mag[idx][idx2]

    return x, y


def export_data(Circuit, size_number, number_of_point, numc):
    """Export all circuits' data into combined x_data and y_data arrays."""
    x = np.zeros((numc, size_number, 3, number_of_point))
    y = np.zeros((numc, size_number))

    for idx in range(numc):
        x[idx], y[idx] = arrange_data(Circuit[idx], idx, size_number, number_of_point)

    x_data = x[0]
    y_data = y[0]

    for idx in range(numc - 1):
        x_data = np.append(x_data, x[idx + 1], axis=0)

    for idx in range(numc - 1):
        y_data = np.append(y_data, y[idx + 1], axis=0)

    return x_data, y_data
