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

#### 4.2.1 Scene Modeling API Example

SHINE extends GalSim's scene building approach by integrating probabilistic programming through numpyro. While GalSim provides deterministic scene generation, SHINE wraps these components in probabilistic distributions to enable Bayesian inference.

**Traditional GalSim Approach:**
```python
# Standard GalSim scene construction
import galsim

# Define galaxy with fixed parameters
gal = galsim.Sersic(n=4, half_light_radius=0.5, flux=1e4)
psf = galsim.Gaussian(sigma=0.1)
final = galsim.Convolve([gal, psf])
image = final.drawImage(scale=0.2)
```

**SHINE Probabilistic Scene Modeling:**
```python
import shine
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random

def shine_scene_model(data=None, config=None):
    """
    Probabilistic scene model that combines GalSim components with numpyro PPL.
    This function defines the generative model for the scene.
    """

    # === Shear Parameters (what we want to infer) ===
    g1 = numpyro.sample('g1', dist.Normal(0., 0.05))
    g2 = numpyro.sample('g2', dist.Normal(0., 0.05))
    shear = galsim.Shear(g1=g1, g2=g2)

    # === PSF Model ===
    psf_sigma = numpyro.sample('psf_sigma',
                                dist.Normal(config['psf_sigma_mean'], 0.01))
    psf = galsim.Gaussian(sigma=psf_sigma)

    # === Scene Configuration ===
    n_galaxies = config.get('n_galaxies', 100)

    # Create empty image with specified dimensions
    image_size = config['image_size']  # pixels
    pixel_scale = config['pixel_scale']  # arcsec/pixel
    image = galsim.ImageF(image_size, image_size, scale=pixel_scale)

    # === Galaxy Population Model ===
    with numpyro.plate('galaxies', n_galaxies):
        # Morphological parameters with hierarchical priors
        sersic_n = numpyro.sample('sersic_n',
                                  dist.TruncatedNormal(2.5, 1.0, low=0.5, high=6.0))

        hlr = numpyro.sample('half_light_radius',
                             dist.LogNormal(jnp.log(0.5), 0.3))

        flux = numpyro.sample('flux',
                              dist.LogNormal(jnp.log(1e4), 0.5))

        # Galaxy ellipticity prior (pre-shear)
        e_intrinsic = numpyro.sample('e_intrinsic',
                                      dist.Beta(2, 5))  # Peaks at low ellipticity

        pa_intrinsic = numpyro.sample('position_angle',
                                       dist.Uniform(0, jnp.pi))

        # Position on detector (in pixels)
        x_pos = numpyro.sample('x_pos',
                                dist.Uniform(1, image_size))
        y_pos = numpyro.sample('y_pos',
                                dist.Uniform(1, image_size))

    # === Render Each Galaxy ===
    # Note: In JAX-GalSim, this would be vectorized
    for i in range(n_galaxies):
        # Build galaxy profile for this object
        galaxy = galsim.Sersic(n=sersic_n[i],
                               half_light_radius=hlr[i],
                               flux=flux[i])

        # Apply intrinsic ellipticity
        galaxy = galaxy.shear(e=e_intrinsic[i],
                             beta=pa_intrinsic[i] * galsim.radians)

        # Apply gravitational shear
        galaxy = galaxy.shear(shear)

        # Convolve with PSF
        final = galsim.Convolve([galaxy, psf])

        # Draw object at specified position
        # Direct drawing with center parameter (accumulates flux for overlapping galaxies)
        final.drawImage(image=image,
                       center=galsim.PositionD(x_pos[i], y_pos[i]),
                       add_to_image=True)

    # === Noise Model ===
    noise_sigma = numpyro.sample('noise_sigma',
                                 dist.HalfNormal(config['noise_estimate']))

    # === Likelihood ===
    if data is not None:
        numpyro.sample('obs',
                       dist.Normal(image, noise_sigma),
                       obs=data)

    return image

# === Configuration Example ===
scene_config = {
    'image_size': 256,
    'pixel_scale': 0.2,  # arcsec/pixel
    'n_galaxies': 100,
    'psf_sigma_mean': 0.1,
    'noise_estimate': 10.0,
    'variable_psf': True,
    'n_psf_basis': 5
}

# === Inference Usage ===
def run_shear_inference(observed_data, scene_config):
    """
    Run HMC to infer shear parameters from observed data.
    """
    from numpyro.infer import MCMC, NUTS

    # Define the kernel
    kernel = NUTS(shine_scene_model)

    # Run MCMC
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4
    )

    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, data=observed_data, config=scene_config)

    # Extract posterior samples
    samples = mcmc.get_samples()

    # Get shear posterior
    g1_posterior = samples['g1']
    g2_posterior = samples['g2']

    return {
        'g1': g1_posterior,
        'g2': g2_posterior,
        'full_samples': samples
    }
```

**Key Features of SHINE's Scene Modeling API:**

1. **Probabilistic Parameters**: All scene parameters can be drawn from probability distributions rather than fixed values
2. **Hierarchical Modeling**: Support for multi-level models (e.g., population-level and individual galaxy parameters)
3. **Full Differentiability**: Uses JAX-GalSim to maintain gradient flow through the entire pipeline
4. **Flexible Priors**: Easy specification of priors on any parameter (shear, galaxy properties, PSF, noise)
5. **Plate Notation**: Leverages numpyro's plate notation for efficient vectorized sampling
6. **Modular Design**: Scene components (galaxies, PSF, noise) are modular and can be customized


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