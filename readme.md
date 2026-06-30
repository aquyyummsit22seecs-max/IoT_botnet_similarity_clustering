# IoT Botnet Clustering and Mitigation using Similarity Measures

This research pipeline has been developed for clustering IoT botnets using similarity measures and mapping behavioral patterns to MITRE ATT&CK techniques.
---

## Overview

This repository implements a six-stage framework for IoT botnet analysis.

Stages include:

1. Feature Extraction
2. Similarity Computation
3. Clustering
4. Validation
5. MITRE ATT&CK Mapping
6. Intelligence Report Generation

---

## Research Objectives

RO1:
Extract behavioral features from IoT BDA analysis dataset artifacts.

RO2:
Clustering botnets using similarity measures, following similarity techniques used,
	• Cosine Similarity	• Jaccard Similarity
Perform Clustering using hierarchical and spectral techniques, and validate
  
RO3:
Mapping security frameowrk such as MITRE ATT&CK for TTPs and Mitigation

---

## Dataset

IoT_BDA Dataset
Contains:
syscalls.json
analysis_results.json
network artifacts
sandbox reports

*Dataset is not included due to size limitations, download dataset from link in dataset readme file and place in its directory.

Place dataset here:
dataset/iot_bda_dataset/tasks/

---

## Repository Structure

src/
dataset/
output/
docs/

---

## Installation

git clone https://github.com/username/IoT-Botnet-Clustering.git

cd IoT-Botnet-Clustering

pip install -r requirements.txt

---

## Run Pipeline

python 0_run_pipeline.py

---

## Pipeline Workflow

Feature Extraction

↓

Cosine Similarity

Jaccard Similarity

↓

Hierarchical Clustering

Spectral Clustering

↓

Validation

↓

MITRE ATT&CK Mapping

↓

Intelligence Report

---

## Outputs

output/features/

output/similarity/

output/clustering/

output/validation/

output/ttp_mitre/

output/intelligence/

---

## Citation

If you use this work please cite:

Author

Title

Year

Institution

