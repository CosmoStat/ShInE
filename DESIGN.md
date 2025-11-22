# Software Design Document

## 1. Introduction

### 1.1 Purpose of the Document
This document describes the architecture and content of the SHINE (SHear INference Environment) software package.

### 1.2 Scope & Objective of the Software
SHINE implements a forward model of gravitational shear in a fully differentiable pipeline using JAX. The software is designed to be applicable to different types of astronomical data (visible and radio) and returns a posterior on shear components. The implementation leverages probabilistic programming language (PPL) tools, particularly numpyro.

## 2. System Overview

### 2.1 High-Level Description of the Software
SHINE provides a forward modeling approach to constrain gravitational shear through the following workflow:
- Generate high-resolution galaxies (using JAX-GalSim, VAE, diffusion models)
- Apply shear and instrumental response (PSF)
- Use the model to infer shear from real data (inference using HMC)

The forward model is built entirely in JAX to ensure full differentiability. It contains all necessary components for application to different types of surveys, including radio and visible surveys such as MeerKAT, EUCLID, and LSST.

### 2.2 Scientific Objectives
The project aims to address limitations in metacalibration, a commonly used tool for shear measurement that:
- Degrades observations
- Only works for weak shear
- Assumes strong knowledge of the PSF

**Target Performance Goals:**
- **Visible surveys:** m ≃ 2 × 10⁻³ and c ≃ 1.5 × 10⁻⁴ (Euclid requirements)
- **Radio surveys:** m ≃ 6.7 × 10⁻³ and c ≃ 8.2 × 10⁻⁴

**Key Advantages:**
- Learn galaxy morphology and properties directly from data
- Bypass ellipticities by directly inferring shear
- Potential to disentangle shear and intrinsic alignments
- Improve photometric redshift estimation

**Challenges and Biases to Address:**
- Model misspecification (realism of image pipeline, galaxy morphology, PSF models and errors)
- Selection/detection biases and blending
- Scalability
- Interface with cosmology
- Automated comparison with current methods

### 2.3 Guiding Principles
- Code must be pip installable on any platform
- Fully written in JAX
- Use JAX-GalSim when possible
- Input via YAML configuration files

### 2.4 Key Features/Capabilities
- Applicable to radio data using JAX-GalSim for galaxy modeling and argosim PSF
- Applicable to Euclid and LSST data with JAX-GalSim for galaxy and PSF modeling
- Support for joint Euclid × LSST analysis


## 3. Design Overview

### 3.1 System Architecture

```
shine/
├── __init__.py
├── main.py
├── config_handler.py
├── scene_modelling/
│   ├── __init__.py
│   └── [modules]
├── inference/
│   ├── __init__.py
│   ├── blackjax.py
│   └── [modules]
├── simulations/
│   ├── __init__.py
│   ├── euclid.py
│   ├── lsst.py
│   └── radio.py
├── data/
│   ├── __init__.py
│   ├── euclid.py
│   ├── lsst.py
│   └── meerkat.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── coverage.py
│   └── comparison.py
├── morphology/
│   ├── __init__.py
│   ├── parametric.py
│   └── non_parametric.py
├── modelling/
│   ├── __init__.py
│   ├── SED.py
│   ├── psf.py
│   └── [modules]
└── wms.py
```

### 3.2 External Interfaces

#### Data Input Interfaces:
- MeerKAT
- Euclid
- LSST

#### Simulation Interfaces:
- Flagship
- CosmoDC2

### 3.3 Third-Party Libraries
- JAX
- JAX-GalSim
- argosim
- numpyro

## 4. Detailed Software Components

### 4.1 Configuration Handler Module
Reads the input YAML configuration file and launches the code.

**Example usage:**
```python
shine.read_input(scene_config, sims_config, data_config,
                 psf_config, forward_model_config, inference_config)
```

### 4.2 Scene Modelling Sub-package
Creates a scene of galaxies with desired properties using a probabilistic model. Includes generation of galaxies, PSF, and noise.

**Input:** Configuration file allowing selection of components (PSF model, galaxy model, etc.). Compatible with GalSim.
**Output:** Log probability function for direct use in numpyro inference, or probability of parameters for external use.


### 4.3 Inference Modelling Sub-package
Contains different modules for various inference methods (MCMC, HMC using blackjax).

**Input:** Log probability
**Output:** Shear posterior

**Example usage:**
```python
.inference(logprob, "HMC", "blackjax")
```

### 4.4 Simulations Input Handler Sub-package
Reads different simulations for use in creating YAML files for the scene modelling module. Each module reads simulation output and converts units to ensure consistency across simulations.

**Input:** Simulation output
**Output:** YAML file

### 4.5 Data Handler Sub-package
Reads real data or simulations used as data. Each module reads survey output and performs unit conversion as needed.

**Input:** Survey output
**Output:** Image

### 4.6 Evaluation Sub-package
Performs evaluation including metrics, coverage tests, and comparison with external techniques.

### 4.7 Morphology Sub-package
Contains parametric and non-parametric morphology generators.

**Input:** YAML file, compatible with GalSim
**Output:** Galaxy profile

### 4.8 Modelling Utilities Sub-package
Contains modules for realistic galaxy scene modeling, particularly components not available in GalSim (e.g., WaveDiff PSF for Euclid).

**Input:** YAML file

### 4.9 Workflow Management System (WMS) Module
Enables job launching on clusters (e.g., SLURM). Supports image segmentation into patches with individual job submission and pipeline status monitoring.

**Input:** YAML file
**Output:** Shell script

### 4.10 Other Extensions
Contains possible extensions such as handling SED and photometric redshift information.

## 5. Development

### 5.1 Development & Testing

#### 5.1.1 Development Strategy
Development begins with a naive pipeline containing simple components, then progresses to ensure robustness across different science cases.

The naive pipeline includes:
- Scene modelling
- Inference
- Simulation output
- Modelling utilities


### 5.2 Style Guide
- **Code style:** Black Python package
- **Documentation:** Use docstrings for every function and Sphinx Python package for HTML documentation generation

### 5.3 Environment
- Use of Docker containers for reproducibility and dependency management