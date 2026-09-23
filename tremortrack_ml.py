"""
TremorTrack
Hybrid Classical-Quantum ML Prototype

Purpose:
    Demonstrates the TremorTrack ML pipeline:
    Sensor Data -> Signal Processing -> Feature Extraction
    -> Classical ML + Quantum ML -> Risk Assessment

NOTE:
    This is a research/prototype implementation.
    It is NOT a medical diagnostic system.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pennylane as qml
from pennylane import numpy as pnp


# ============================================================
# 1. SIGNAL PROCESSING
# ============================================================

def calculate_magnitude(ax, ay, az):
    """
    Calculate acceleration magnitude from 3-axis data.
    """
    return np.sqrt(ax**2 + ay**2 + az**2)


def remove_background(magnitude):
    """
    Simple background/gravity removal.
    """
    return magnitude - np.mean(magnitude)


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(ax, ay, az, sampling_rate=50):
    """
    Extract simple movement features from accelerometer data.
    """

    magnitude = calculate_magnitude(ax, ay, az)
    signal = remove_background(magnitude)

    # RMS movement energy
    rms = np.sqrt(np.mean(signal ** 2))

    # Signal variability
    std = np.std(signal)

    # Maximum movement
    peak = np.max(np.abs(signal))

    # Frequency analysis
    fft_values = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_rate)

    # Ignore DC component
    if len(fft_values) > 1:
        dominant_frequency = frequencies[
            np.argmax(fft_values[1:]) + 1
        ]
    else:
        dominant_frequency = 0

    return [
        rms,
        std,
        peak,
        dominant_frequency
    ]


# ============================================================
# 3. CREATE DEMO DATASET
# ============================================================

def generate_demo_dataset(samples=200):

    np.random.seed(42)

    features = []
    labels = []

    for i in range(samples):

        # Simulated accelerometer data
        time = np.linspace(0, 5, 250)

        ax = np.random.normal(0, 0.08, 250)
        ay = np.random.normal(0, 0.08, 250)
        az = np.random.normal(1, 0.08, 250)

        # Simulate stronger periodic movement
        if i >= samples // 2:

            tremor_frequency = np.random.uniform(3, 7)

            tremor = (
                0.25
                * np.sin(
                    2 * np.pi
                    * tremor_frequency
                    * time
                )
            )

            ax += tremor

            label = 1

        else:
            label = 0

        feature_vector = extract_features(
            ax,
            ay,
            az
        )

        features.append(feature_vector)
        labels.append(label)

    return np.array(features), np.array(labels)


# ============================================================
# 4. CLASSICAL MACHINE LEARNING
# ============================================================

def train_classical_model(X_train, X_test, y_train, y_test):

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== CLASSICAL ML ==========")
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return model, accuracy


# ============================================================
# 5. QUANTUM MACHINE LEARNING
# ============================================================

N_QUBITS = 4

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):

    # Encode classical features into quantum circuit
    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    # Trainable quantum layers
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return qml.expval(
        qml.PauliZ(0)
    )


def train_quantum_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    # Convert labels:
    # 0 -> -1
    # 1 -> +1
    y_train_q = 2 * y_train - 1
    y_test_q = 2 * y_test - 1

    # Convert features to quantum-compatible format
    X_train_q = pnp.array(
        X_train,
        requires_grad=False
    )

    y_train_q = pnp.array(
        y_train_q,
        requires_grad=False
    )

    # Quantum circuit weights
    rng = np.random.default_rng(42)

    weights = pnp.array(
        0.01 * rng.normal(
            size=(2, N_QUBITS, 3)
        ),
        requires_grad=True
    )

    optimizer = qml.AdamOptimizer(
        stepsize=0.05
    )

    def cost_fn(weights):

        predictions = pnp.stack([
            quantum_circuit(
                X_train_q[i],
                weights
            )
            for i in range(len(X_train_q))
        ])

        return pnp.mean(
            (predictions - y_train_q) ** 2
        )

    print("\n========== QUANTUM ML ==========")
    print("Training Variational Quantum Classifier...")

    epochs = 30

    for epoch in range(epochs):

        weights, cost = optimizer.step_and_cost(
            cost_fn,
            weights
        )

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {float(cost):.4f}"
            )

    # Test quantum model
    predictions = []

    for sample in X_test:

        output = quantum_circuit(
            pnp.array(sample),
            weights
        )

        predicted_class = (
            1 if output >= 0 else 0
        )

        predictions.append(
            predicted_class
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nVQC Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    return weights, accuracy


# ============================================================
# 6. RISK ASSESSMENT
# ============================================================

def risk_assessment(
    classical_model,
    quantum_weights,
    scaler,
    features
):

    scaled_features = scaler.transform(
        [features]
    )[0]

    # Classical prediction
    classical_probability = (
        classical_model.predict_proba(
            [scaled_features]
        )[0][1]
    )

    # Quantum prediction
    quantum_output = quantum_circuit(
        pnp.array(scaled_features),
        quantum_weights
    )

    quantum_probability = (
        float(quantum_output) + 1
    ) / 2

    # Combined score
    combined_score = (
        classical_probability
        + quantum_probability
    ) / 2

    if combined_score >= 0.70:
        level = "Higher-risk pattern"

    elif combined_score >= 0.40:
        level = "Moderate-risk pattern"

    else:
        level = "Lower-risk pattern"

    return {
        "Classical Score": round(
            classical_probability, 3
        ),
        "Quantum Score": round(
            quantum_probability, 3
        ),
        "Combined Score": round(
            combined_score, 3
        ),
        "Assessment": level
    }


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("TREMORTRACK - HYBRID QUANTUM-CLASSICAL ML")
    print("=" * 60)

    # Generate prototype dataset
    X, y = generate_demo_dataset()

    print("\nDataset created")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Normalize features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # Classical ML
    classical_model, classical_accuracy = (
        train_classical_model(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # Quantum ML
    quantum_weights, quantum_accuracy = (
        train_quantum_model(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # Model comparison
    print("\n========== MODEL COMPARISON ==========")

    print(
        f"Classical SVM : "
        f"{classical_accuracy * 100:.2f}%"
    )

    print(
        f"Quantum VQC   : "
        f"{quantum_accuracy * 100:.2f}%"
    )

    # Example new measurement
    new_ax = np.random.normal(
        0, 0.12, 250
    )

    new_ay = np.random.normal(
        0, 0.12, 250
    )

    new_az = (
        np.random.normal(
            1, 0.08, 250
        )
    )

    # Extract features
    new_features = extract_features(
        new_ax,
        new_ay,
        new_az
    )

    # Risk assessment
    result = risk_assessment(
        classical_model,
        quantum_weights,
        scaler,
        new_features
    )

    print("\n========== RISK ASSESSMENT ==========")

    for key, value in result.items():
        print(
   """
TremorTrack
Hybrid Classical-Quantum ML Prototype

Purpose:
    Demonstrates the TremorTrack ML pipeline:
    Sensor Data -> Signal Processing -> Feature Extraction
    -> Classical ML + Quantum ML -> Risk Assessment

NOTE:
    This is a research/prototype implementation.
    It is NOT a medical diagnostic system.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pennylane as qml
from pennylane import numpy as pnp


# ============================================================
# 1. SIGNAL PROCESSING
# ============================================================

def calculate_magnitude(ax, ay, az):
    """
    Calculate acceleration magnitude from 3-axis data.
    """
    return np.sqrt(ax**2 + ay**2 + az**2)


def remove_background(magnitude):
    """
    Simple background/gravity removal.
    """
    return magnitude - np.mean(magnitude)


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(ax, ay, az, sampling_rate=50):
    """
    Extract simple movement features from accelerometer data.
    """

    magnitude = calculate_magnitude(ax, ay, az)
    signal = remove_background(magnitude)

    # RMS movement energy
    rms = np.sqrt(np.mean(signal ** 2))

    # Signal variability
    std = np.std(signal)

    # Maximum movement
    peak = np.max(np.abs(signal))

    # Frequency analysis
    fft_values = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_rate)

    # Ignore DC component
    if len(fft_values) > 1:
        dominant_frequency = frequencies[
            np.argmax(fft_values[1:]) + 1
        ]
    else:
        dominant_frequency = 0

    return [
        rms,
        std,
        peak,
        dominant_frequency
    ]


# ============================================================
# 3. CREATE DEMO DATASET
# ============================================================

def generate_demo_dataset(samples=200):

    np.random.seed(42)

    features = []
    labels = []

    for i in range(samples):

        # Simulated accelerometer data
        time = np.linspace(0, 5, 250)

        ax = np.random.normal(0, 0.08, 250)
        ay = np.random.normal(0, 0.08, 250)
        az = np.random.normal(1, 0.08, 250)

        # Simulate stronger periodic movement
        if i >= samples // 2:

            tremor_frequency = np.random.uniform(3, 7)

            tremor = (
                0.25
                * np.sin(
                    2 * np.pi
                    * tremor_frequency
                    * time
                )
            )

            ax += tremor

            label = 1

        else:
            label = 0

        feature_vector = extract_features(
            ax,
            ay,
            az
        )

        features.append(feature_vector)
        labels.append(label)

    return np.array(features), np.array(labels)


# ============================================================
# 4. CLASSICAL MACHINE LEARNING
# ============================================================

def train_classical_model(X_train, X_test, y_train, y_test):

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== CLASSICAL ML ==========")
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return model, accuracy


# ============================================================
# 5. QUANTUM MACHINE LEARNING
# ============================================================

N_QUBITS = 4

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):

    # Encode classical features into quantum circuit
    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    # Trainable quantum layers
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return qml.expval(
        qml.PauliZ(0)
    )


def train_quantum_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    # Convert labels:
    # 0 -> -1
    # 1 -> +1
    y_train_q = 2 * y_train - 1
    y_test_q = 2 * y_test - 1

    # Convert features to quantum-compatible format
    X_train_q = pnp.array(
        X_train,
        requires_grad=False
    )

    y_train_q = pnp.array(
        y_train_q,
        requires_grad=False
    )

    # Quantum circuit weights
    rng = np.random.default_rng(42)

    weights = pnp.array(
        0.01 * rng.normal(
            size=(2, N_QUBITS, 3)
        ),
        requires_grad=True
    )

    optimizer = qml.AdamOptimizer(
        stepsize=0.05
    )

    def cost_fn(weights):

        predictions = pnp.stack([
            quantum_circuit(
                X_train_q[i],
                weights
            )
            for i in range(len(X_train_q))
        ])

        return pnp.mean(
            (predictions - y_train_q) ** 2
        )

    print("\n========== QUANTUM ML ==========")
    print("Training Variational Quantum Classifier...")

    epochs = 30

    for epoch in range(epochs):

        weights, cost = optimizer.step_and_cost(
            cost_fn,
            weights
        )

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {float(cost):.4f}"
            )

    # Test quantum model
    predictions = []

    for sample in X_test:

        output = quantum_circuit(
            pnp.array(sample),
            weights
        )

        predicted_class = (
            1 if output >= 0 else 0
        )

        predictions.append(
            predicted_class
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nVQC Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    return weights, accuracy


# ============================================================
# 6. RISK ASSESSMENT
# ============================================================

def risk_assessment(
    classical_model,
    quantum_weights,
    scaler,
    features
):

    scaled_features = scaler.transform(
        [features]
    )[0]

    # Classical prediction
    classical_probability = (
        classical_model.predict_proba(
            [scaled_features]
        )[0][1]
    )

    # Quantum prediction
    quantum_output = quantum_circuit(
        pnp.array(scaled_features),
        quantum_weights
    )

    quantum_probability = (
        float(quantum_output) + 1
    ) / 2

    # Combined score
    combined_score = (
        classical_probability
        + quantum_probability
    ) / 2

    if combined_score >= 0.70:
        level = "Higher-risk pattern"

    elif combined_score >= 0.40:
        level = "Moderate-risk pattern"

    else:
        level = "Lower-risk pattern"

    return {
        "Classical Score": round(
            classical_probability, 3
        ),
        "Quantum Score": round(
            quantum_probability, 3
        ),
        "Combined Score": round(
            combined_score, 3
        ),
        "Assessment": level
    }


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("TREMORTRACK - HYBRID QUANTUM-CLASSICAL ML")
    print("=" * 60)

    # Generate prototype dataset
    X, y = generate_demo_dataset()

    print("\nDataset created")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Normalize features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # Classical ML
    classical_model, classical_accuracy = (
        train_classical_model(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # Quantum ML
    quantum_weights, quantum_accuracy = (
        train_quantum_model(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # Model comparison
    print("\n========== MODEL COMPARISON ==========")

    print(
        f"Classical SVM : "
        f"{classical_accuracy * 100:.2f}%"
    )

    print(
        f"Quantum VQC   : "
        f"{quantum_accuracy * 100:.2f}%"
    )

    # Example new measurement
    new_ax = np.random.normal(
        0, 0.12, 250
    )

    new_ay = np.random.normal(
        0, 0.12, 250
    )

    new_az = (
        np.random.normal(
            1, 0.08, 250
        )
    )

    # Extract features
    new_features = extract_features(
        new_ax,
        new_ay,
        new_az
    )

    # Risk assessment
    result = risk_assessment(
        classical_model,
        quantum_weights,
        scaler,
        new_features
    )

    print("\n========== RISK ASSESSMENT ==========")

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("\nNOTE:")
    print(
        "This prototype provides a research-oriented "
        "risk assessment only."
    )

    print(
        "It must not be used as a standalone "
        "medical diagnosis."
    )


if __name__ == "__main__":
    main()         f"{key}: {value}"
        )

    print("\nNOTE:")
    print(
        "This prototype provides a research-oriented "
        "risk assessment only."
    )

    print(
        "It must not be used as a standalone "
        "medical diagnosis."
    )


if __name__ == "__main__":
    main()
