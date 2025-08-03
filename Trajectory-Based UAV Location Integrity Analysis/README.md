# Trajectory-Based UAV Location Integrity Analysis

This repository contains a set of Jupyter notebooks that implement the data preprocessing and transformation pipeline used in our trajectory-driven deep learning approach for UAV location integrity checks.

The work is based on the research published in **IEEE Access (2024)**:  
**"Trajectory-Driven Deep Learning for UAV Location Integrity Checks"**  
DOI: [10.1109/ACCESS.2024.3507637](https://doi.org/10.1109/ACCESS.2024.3507637)

---

## 📌 Overview

The goal of this project is to detect GPS spoofing and location falsification attacks in UAVs **without using GPS signal-specific features**, instead relying on **movement-based trajectory features**. The project builds a machine learning pipeline that transforms raw UAV flight logs into structured sequences suitable for RNN and Transformer-based models.

---

## 🧱 Notebook Structure & Version History

Each notebook corresponds to a stage in the research development process, showing incremental improvements in data preprocessing and representation.

| Version | Notebook                         | Description |
|--------|----------------------------------|-------------|
| `v2`   | `01_dataset_preprocessing.ipynb` | Basic data loading and preprocessing. This version implements the initial cleaning and organization of raw UAV flight data (CSV format). Includes class labeling and partitioning into clean vs. attack samples. |
| `v7`   | `02_sequence_data_builder.ipynb` | Implements **sequence construction** using a sliding window over timestamped flight records. Converts tabular data into fixed-length sequences to enable time-series modeling. |
| `v8`   | `03_transformer_input_builder.ipynb` | Focuses on Transformer-specific data handling. Adds normalization and trajectory-aware features such as positional deltas, Euclidean distance, and velocity vectors for attention-based models. |
| `v9`   | `04_full_sequence_pipeline.ipynb` | Full preprocessing pipeline including sequence transformation, feature engineering, and model-ready input preparation. Integrates all prior improvements and serves as the final data builder for model training. |

---

## ✨ Key Features

- Fully GPS-agnostic: no reliance on jamming/noise/signal quality fields.
- Custom sequence generation (sliding window approach).
- Feature engineering for trajectory modeling (ΔPosition, ΔTime, Speed, Direction).
- Support for both RNN-based and Transformer-based models.
- Modular and reproducible code for research and deployment.

---

## 🔧 Requirements

This code is intended to be run in a Python 3.8+ environment with:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `torch` (for modeling; not included in notebooks here)

---

## 📄 Reference

Please cite the following publication if you use or reference this work:

> M. Shin, S.-Y. Chang, J. Kim, K. Park, and J. Kim,  
> **"Trajectory-Driven Deep Learning for UAV Location Integrity Checks"**,  
> *IEEE Access*, 2024. DOI: [10.1109/ACCESS.2024.3507637](https://doi.org/10.1109/ACCESS.2024.3507637)

---

## 📬 Contact

This repository was created as part of a research project during my graduate program.  
For questions or collaboration opportunities, feel free to reach out via GitHub or email.

