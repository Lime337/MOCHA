# MOCHA

**Modular Continual Hyperdimensional Adaptation**

MOCHA is a unified Hyperdimensional Computing (HDC) framework for lightweight learning, adaptation, and reversible model updates across heterogeneous edge-intelligence applications.

This repository contains the implementations associated with three application domains:

- **HERMES** — Android malware detection
- **POSEiDON** — Human pose classification
- **THEMIS** — Wearable activity recognition and personalization

The repository also includes a lightweight interactive demonstration of the MOCHA framework.

> **Note:** This repository accompanies ongoing research. The current graphical demo is intended as a simple visualization of the MOCHA framework and does not reproduce the full experimental evaluation.

---

## Overview

Hyperdimensional Computing represents information using high-dimensional vectors, or hypervectors, and performs learning through lightweight algebraic operations.

MOCHA uses application-specific HDC encoders while maintaining a common prototype-based learning interface:

```text
Domain Input
     |
     v
Application-Specific Encoder
     |
     v
Bipolar Hypervector
     |
     v
Class Prototype
     |
     v
Similarity-Based Prediction
```

The three applications use different input representations and encoding strategies while sharing the same general HDC learning principles.

### HERMES

HERMES applies HDC to Android malware detection.

```text
APK / Opcode Data
        |
        v
Opcode + Method Encoding
        |
        v
Hypervector
        |
        v
Benign / Malware Classification
```

### POSEiDON

POSEiDON applies HDC to human pose classification using body landmarks, spatial information, and joint-angle representations.

```text
Pose Landmarks
      |
      v
Spatial + Angle Encoding
      |
      v
Hypervector
      |
      v
Pose Classification
```

### THEMIS

THEMIS applies HDC to wearable sensor-based activity recognition and supports lightweight personalization and reversible updates.

```text
Wearable Sensor Window
          |
          v
Temporal Sensor Encoding
          |
          v
Hypervector
          |
          v
Activity Classification
```

---

## Reversible Adaptation

A central capability demonstrated by MOCHA is the ability to modify HDC class prototypes using additive operations.

A contribution can be added to a class prototype as

```text
P_c <- P_c + h
```

where `P_c` is the class prototype and `h` is an encoded sample hypervector.

If the contribution is later determined to be incorrect or undesirable, the registered contribution can be algebraically removed:

```text
P_c <- P_c - h
```

This allows previously registered contributions to be reversed without reconstructing the entire model from scratch.

---

## Repository Structure

```text
MOCHA/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- HERMES/
|   |-- build_class_FAST.py
|   |-- inference.py
|   |-- hdc_newdata.py
|   `-- HERMES_MOCHA_unlearning.py
|
|-- POSEiDON/
|   |-- angle_hdc_dict.py
|   |-- angle_hdc_encoder.py
|   |-- hcd_pose_dictionary.py
|   |-- hybrid_hdc_encoder.py
|   |-- hybrid_hdc_training.py
|   `-- POSEiDON_MOCHA_Sobol.py
|
`-- THEMIS/
    |-- HDC_exp1.py
    |-- HDC_exp2.py
    |-- HDC_exp3.py
    |-- HDC_exp4.py
    |-- HDC_exp5.py
    |-- HDC_exp5.1.py
    |-- HDC_exp6.py
    |-- HDC_exp6_appended_noise.py
    `-- HDC_reversible_stability.py
```

---

## Interactive MOCHA Demo

The repository includes a lightweight Streamlit interface that visually demonstrates the common MOCHA workflow.

The demo allows the user to select between:

- HERMES
- POSEiDON
- THEMIS

For each application, the interface displays its domain-specific input and encoding pipeline while demonstrating the common HDC representation and reversible prototype update mechanism.

The demo includes three simple operations:

### Inject Bad Data

Simulates an incorrect contribution being added to a class prototype.

```text
P_c <- P_c + h_bad
```

### Unlearn

Simulates algebraic removal of the incorrect contribution.

```text
P_c <- P_c - h_bad
```

### Reset

Returns the visualization to its original state.

The displayed hypervector is intentionally reduced to a small number of dimensions for visualization. Actual HDC models operate in much higher-dimensional spaces.

The accuracy values displayed by the graphical demo are **illustrative only** and should not be interpreted as experimental results.

---

## Running the Demo

### 1. Clone the repository

```bash
git clone <repository-url>
cd MOCHA
```

### 2. Install the demo dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit interface

```bash
python -m streamlit run app.py
```

Streamlit should automatically open the interface in your default web browser.

If it does not open automatically, navigate to the local address displayed in the terminal, typically:

```text
http://localhost:8501
```

---

## Requirements

The graphical demonstration requires:

- Python
- Streamlit
- NumPy

Additional research scripts may require packages including:

- pandas
- SciPy
- scikit-learn
- PyTorch
- Androguard

Exact dependencies vary between HERMES, POSEiDON, and THEMIS.

---

## Datasets

The research datasets are **not distributed directly with this repository**.

The experimental scripts may require externally obtained datasets and locally generated intermediate files.

Examples include Android malware datasets, pose landmark data, and wearable activity data.

Users should obtain the appropriate datasets from their original sources and follow the corresponding preprocessing requirements before attempting to reproduce the full research experiments.

The interactive `app.py` demonstration does **not** require these research datasets.

---

## Research Code

The HERMES, POSEiDON, and THEMIS directories contain research-oriented implementations and experimental scripts associated with the MOCHA framework.

These scripts are retained separately from the graphical demonstration so that the lightweight demo does not depend on large datasets, trained models, or application-specific preprocessing pipelines.

Some research scripts may require configuration of local dataset paths before execution.

---

## Disclaimer

The Streamlit application is intended to provide a conceptual visualization of MOCHA's HDC processing and reversible adaptation mechanisms.

The graphical demo does not execute the complete HERMES, POSEiDON, or THEMIS experimental pipelines. Demonstration values and the displayed low-dimensional hypervectors are illustrative.

For experimental methodology and quantitative results, please refer to the associated research publication.

---

## Citation

Citation information for the MOCHA publication will be added following publication.

```bibtex
@article{mocha,
  title   = {MOCHA: Modular Continual Hyperdimensional Adaptation},
  author  = {Citation information to be added},
  journal = {To appear},
  year    = {2026}
}
```

---

## License

License information will be added prior to the public release of the repository.

---

## Authors

This repository contains research code developed as part of the MOCHA framework.

Additional author and affiliation information will be added with the final public release.