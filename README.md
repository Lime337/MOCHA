# MOCHA

**Modular Continual Hyperdimensional Adaptation for Secure Edge Intelligence**

MOCHA is a research framework for lightweight, adaptive Hyperdimensional Computing (HDC) on edge devices. It integrates three application-specific encoders through a common interface and shared prototype-based adaptation principles, including additive learning, reversible sample updates, and similarity-based classification.

This repository accompanies the 2026 paper **“MOCHA: Modular Continual Hyperdimensional Adaptation for Secure Edge Intelligence”** in *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems* (IEEE TCAD).

**Paper:** [IEEE Xplore — Document 11806764](https://ieeexplore.ieee.org/document/11806764)

## Framework

The framework retains domain-specific encoding, rather than imposing a single feature representation on every application. Each encoder produces a hypervector and the corresponding information required by a common MOCHA interface. The shared adaptation core maintains class prototype states and supports learning, unlearning, source removal or replacement, and similarity-based inference.

| Component | Application | Representation |
| --- | --- | --- |
| **HERMES** | Android malware detection | Opcode and method information encoded as HDC representations |
| **POSEiDON** | Human pose classification | Landmark identity, spatial position, and joint-angle representations |
| **THEMIS** | Wearable activity recognition | Sensor features and temporal information for classification and personalization |

HERMES and POSEiDON originate from earlier domain-specific work. **MOCHA's unified framework, reversible adaptation extensions, and THEMIS wearable sensing evaluation are contributions of the TCAD study.** Refer to the paper for the precise distinction between prior and newly reported experiments.

### Reversible learning

For a retained (non-thresholded) class state \(P_c\), an encoded contribution \(h\) can be added or subtracted:

```text
Learn:    P_c <- P_c + h
Unlearn:  P_c <- P_c - h
```

More generally, weighted or reassigned updates must be reversed using the same contribution and weight originally applied. This requires retaining sufficient update information; unlearning is not generally recoverable from a binarized class vector alone. The paper describes the adaptation mechanisms and evaluation protocols in detail.

## Repository layout

```text
MOCHA/
├── app.py                        # Interactive conceptual demo
├── requirements.txt              # Lightweight demo dependencies
├── requirements-research.txt     # Additional research-script dependencies
├── HERMES/                       # Android malware experiments and unlearning
├── POSEiDON/                     # Pose encoders, training, and MOCHA evaluation
└── THEMIS/                       # Wearable sensing and adaptation experiments
    ├── HDC_exp1.py               # Baseline encoding/classification
    ├── HDC_exp2.py               # Personalization
    ├── HDC_exp3.py               # Reversal of personalization updates
    ├── HDC_exp4.py               # Corrupted updates and unlearning
    ├── HDC_exp5.py               # Corrupted-sensor contribution removal
    ├── HDC_exp5.1.py             # Approximate sensor-key analysis
    ├── HDC_exp6.py               # Sensor-group ablation with injected faults
    ├── HDC_exp6_appended_noise.py # Appended noisy sensor-group ablation
    └── HDC_reversible_stability.py
```

Additional scripts in each subdirectory implement the corresponding encoding, training, classification, or evaluation stages. Filenames and experiment-specific configuration constants are retained for traceability to the research code.

## Interactive demonstration

The [Streamlit](https://streamlit.io/) application in `app.py` presents the shared MOCHA workflow for HERMES, POSEiDON, and THEMIS. It illustrates prototype changes after injecting an erroneous contribution, removing that contribution, and resetting the display.

**The demo is conceptual:** it does not run the complete experiments or load trained research models. Its displayed accuracy values and low-dimensional hypervectors are illustrative, **not** measurements reported in the paper.

### Run the demo

1. Use Python 3 and open a terminal in the repository directory.
2. Install the lightweight dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the interface:

   ```bash
   python -m streamlit run app.py
   ```

Open the local URL printed by Streamlit if your browser does not open automatically (typically `http://localhost:8501`). The demo does not require external datasets.

## Research scripts and datasets

The research scripts require additional packages, datasets, and sometimes pretrained or intermediate files. You can install commonly referenced Python dependencies with:

```bash
python -m pip install -r requirements-research.txt
```

Depending on the experiment, additional platform-specific setup or compatible package versions may be necessary. Some scripts use local paths or output artifacts generated by preceding experiments; review the configuration constants at the top of the script before running it. `requirements-research.txt` is a dependency guide rather than a fully pinned, validated reproducibility environment.

**Datasets and model artifacts are not included in this repository.** Relevant research inputs include:

- **HERMES:** Android application samples, opcode/method features, and AndroZoo/AndMal evaluation data.
- **POSEiDON:** Human pose landmark recordings, pose dictionaries, and the custom/YOGA-5 evaluation inputs.
- **THEMIS:** PAMAP2 wearable activity measurements, windowed `.npz` caches, and saved encoder or leave-one-subject-out model files generated by the experiment pipeline.

Obtain any necessary datasets from the original providers and respect their access conditions. Experimental results in the publication depend on the paper's stated preprocessing, evaluation splits, and configurations; running individual scripts with different local settings is not equivalent to reproducing those results.

## Results and scope

The TCAD paper reports malware classification, pose recognition, wearable personalization, reversible unlearning, robustness analyses, and edge deployment measurements. For authoritative numerical results, hardware conditions, comparisons, and experimental limitations, consult the [published paper](https://ieeexplore.ieee.org/document/11806764).

This repository provides the component-level research implementations and a framework illustration. It should not be interpreted as a single end-to-end, install-and-run reproduction package for every table or figure in the publication.

## Citation

If you use this repository, please cite:

> E. Meyer, T. Tasfia, A. Masum, C. Dupuis, and S. Aygun, “MOCHA: Modular Continual Hyperdimensional Adaptation for Secure Edge Intelligence,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2026. [IEEE Xplore](https://ieeexplore.ieee.org/document/11806764).

Check the IEEE Xplore record for the final indexed bibliographic metadata, including volume, issue, pages, and DOI.

## Authors

Emilien Meyer, Tanha Tasfia, Abu Masum, Colin Dupuis, and Sercan Aygun  
University of Louisiana at Lafayette, Louisiana, USA

## License

No open-source license is included in this repository. Until the authors specify one, permission to use, modify, or redistribute the code is not granted by default. Please contact the authors for licensing questions.
