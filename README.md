# GS2E: Gaussian Splatting for Event Stream Generation

## Overview

This repository provides an implementation of **GS2E (Gaussian Splatting is an Effective Data Generator for Event Stream Generation)**.

GS2E leverages **3D Gaussian Splatting (3DGS)** to construct controllable and photorealistic image sequences that can subsequently be converted into event streams. The framework is designed to support research in event-based vision by providing a computational pipeline for generating and evaluating event-stream data.

The repository contains the implementation of the Gaussian rendering pipeline, scene processing utilities, event-stream generation components, training and rendering scripts, evaluation utilities, and associated resources.

## Key Components

The repository includes the following major components:

* **3D Gaussian Splatting**

  * Gaussian representation and scene processing
  * Differentiable rendering components
  * Gaussian-based scene reconstruction and rendering

* **Event Stream Generation**

  * Generation of event streams from rendered image sequences
  * Event-stream output and storage
  * Support for subsequent event-based vision experiments

* **Rendering and Processing**

  * Camera and trajectory processing
  * Image rendering
  * Scene conversion and preprocessing

* **Evaluation**

  * Quantitative evaluation utilities
  * Image and rendering quality metrics
  * Experimental result documentation

## Repository Structure

```text
GS2E/
│
├── arguments/              # Argument and configuration definitions
├── assets/                 # Supporting assets and resources
├── event_stream_output/    # Generated event-stream outputs
├── gaussian_renderer/      # Gaussian Splatting renderer
├── lpipsPyTorch/           # LPIPS implementation
├── output/                 # Training and rendering outputs
│   └── tandt_train/
├── scene/                  # Scene representation and processing
├── utils/                  # Utility functions
│
├── convert.py              # Scene/data conversion utilities
├── train.py                # Gaussian Splatting training
├── render.py               # Scene rendering
├── gs2e_step4.py           # GS2E processing pipeline
├── gs2e_step5.py           # GS2E processing pipeline
├── full_eval.py            # Full evaluation pipeline
├── metrics.py              # Evaluation metrics
├── environment.yml         # Environment and dependency specification
├── results.md              # Experimental results
├── LICENSE.md              # License information
└── README.md               # Project documentation
```

The repository currently contains the above implementation components and supporting resources.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Pragati-Nayak23/GS2E.git
cd GS2E
```

### 2. Create the Environment

The repository provides an `environment.yml` file containing the environment specification.

```bash
conda env create -f environment.yml
conda activate gs2e
```

If the environment name differs from the one specified in the environment file, activate the corresponding environment name defined there.

### 3. Verify the Installation

After installing the dependencies, verify that the required Python packages and the GPU-enabled PyTorch environment are correctly configured.

For GPU-based experiments, ensure that the installed CUDA, PyTorch, and NVIDIA driver versions are mutually compatible.

## Usage

The repository provides separate scripts for the major stages of the pipeline.

### Training

The Gaussian Splatting scene can be trained using:

```bash
python train.py
```

### Rendering

After training, rendered image sequences can be generated using:

```bash
python render.py
```

### GS2E Pipeline

The repository contains dedicated processing stages for the GS2E pipeline:

```bash
python gs2e_step4.py
python gs2e_step5.py
```

The exact arguments and input/output paths should be configured according to the experimental setup.

### Evaluation

The complete evaluation pipeline can be executed using:

```bash
python full_eval.py
```

Additional evaluation metrics are implemented in:

```text
metrics.py
```

## Experimental Results

Experimental results and evaluation information are documented in `results.md`.

The repository includes evaluation-related components for assessing the quality of the generated results and the underlying Gaussian Splatting implementation.

## Pipeline

The overall GS2E workflow can be summarized as:

```text
Input Scene
     |
     v
3D Gaussian Splatting
     |
     v
Scene Reconstruction
     |
     v
Camera / Trajectory Processing
     |
     v
Image Sequence Rendering
     |
     v
Event Stream Generation
     |
     v
Event Stream Output
     |
     v
Evaluation and Analysis
```

This pipeline enables rendered image sequences to be used as the basis for generating event-stream data for event-based vision research.

## Research Applications

The implementation can be used as a foundation for research involving:

* Event-based computer vision
* Event camera simulation
* Synthetic event-stream dataset generation
* 3D Gaussian Splatting
* Novel-view rendering
* Camera trajectory simulation
* Vision-based learning with event data
* Multimodal RGB-event vision
* Evaluation of event-based vision algorithms

## Relation to the Original GS2E Work

This repository is based on the GS2E research framework described in:

> **GS2E: Gaussian Splatting is an Effective Data Generator for Event Stream Generation**

The original GS2E work introduces a pipeline that uses 3D Gaussian Splatting to manipulate camera trajectories, render image sequences, and generate simulated event streams.

Users interested in the underlying methodology should refer to the original publication and its official implementation.

## Citation

If you use this repository, its source code, implementation, generated data, algorithms, experimental setup, or any substantial component of this work in an academic, research, educational, or public project, **please cite the associated GS2E work and this repository appropriately**.

### GS2E Publication

```bibtex
@misc{li2025gs2egaussiansplattingeffective,
  title={GS2E: Gaussian Splatting is an Effective Data Generator for Event Stream Generation},
  author={Yuchen Li and Chaoran Feng and Zhenyu Tang and Kaiyuan Deng and Wangbo Yu and Yonghong Tian and Li Yuan},
  year={2025},
  eprint={2505.15287},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.15287}
}
```

The publication metadata above is from the official GS2E implementation/repository.

### Repository Citation

If you specifically use this repository or modifications maintained here, you may additionally cite:

```bibtex
@misc{nayak_gs2e,
  author       = {Nayak, Pragati},
  title        = {GS2E},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Pragati-Nayak23/GS2E}
}
```

## Attribution and Responsible Use

If this repository is used in a research publication, thesis, technical report, benchmark, dataset, or publicly released software, please provide appropriate attribution to the original GS2E research work and clearly identify any modifications made to this implementation.

If results obtained using this repository are reported publicly, users are encouraged to document the relevant configuration, dataset, scene, rendering settings, and evaluation protocol to support reproducibility.

## Contact

For technical questions, research-related inquiries, collaboration, or issues concerning this repository, please contact:

**Pragati Nayak**
Email: [pragati23@iiserb.ac.in](mailto:pragati23@iiserb.ac.in)

## Acknowledgements

This work builds upon the broader research and software ecosystem surrounding **3D Gaussian Splatting** and event-based vision.

We acknowledge the authors and developers of the underlying open-source projects and research contributions that make this implementation possible.


[GS2E Repository](https://github.com/Pragati-Nayak23/GS2E)
