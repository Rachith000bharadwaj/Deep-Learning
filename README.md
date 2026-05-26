# Deep Learning Project: CvT vs ConvNeXT

This repository contains a Deep Learning project comparing two modern vision model families: Convolutional Vision Transformer (CvT) and ConvNeXT. The project focuses on image classification experiments, training results, evaluation metrics, visual analysis, and model comparison.

The work is organized around the question of how transformer-inspired vision models and modern convolutional architectures perform on image classification tasks when compared through accuracy, loss, speed, checkpoint behavior, and visualization outputs.

## Project Overview

The project compares:

- **CvT (Convolutional Vision Transformer)**: a vision transformer style architecture that combines convolutional operations with transformer-based representation learning.
- **ConvNeXT / ConvNeXT V2**: a modernized convolutional neural network architecture inspired by transformer-era design improvements.

The repository includes training notebooks, experiment outputs, metric files, visualization figures, model summaries, and a final report.

## Repository Structure

```text
.
├── CvT vs ConvNeXT/
│   ├── all_model_training_testing_optimization.ipynb
│   ├── build_final_report_notebook.py
│   ├── convnext/
│   │   ├── history.csv
│   │   ├── history.json
│   │   ├── metrics.csv
│   │   ├── requirements.txt
│   │   ├── summary.json
│   │   ├── inaturalist_optimization_status.json
│   │   ├── convnextv2_tiny_food101_10epoch_b16/
│   │   ├── smoke_outputs/
│   │   └── smoke_outputs_2/
│   ├── cvt/
│   │   ├── cvt13_compile_best.pth
│   │   ├── cvt13_compile_bs64_best.pth
│   │   ├── cvt13_compile_sdpa_math_bs64_best.pth
│   │   ├── cvt13_food101_best.pth
│   │   └── metric JSON files
│   └── figures/
│       ├── food-101/
│       └── inaturalist/
├── Report.pdf
├── .gitignore
└── README.md
```

## Main Components

### 1. Training and Experiment Notebook

`CvT vs ConvNeXT/all_model_training_testing_optimization.ipynb`

This notebook contains the main experiment workflow. It covers training, testing, optimization, metric collection, and comparison between CvT and ConvNeXT-based approaches.

### 2. ConvNeXT Experiments

`CvT vs ConvNeXT/convnext/`

This folder contains ConvNeXT experiment outputs, including:

- Training history files
- Metrics files
- Summary files
- iNaturalist optimization status
- Smoke-test output folders
- ConvNeXT V2 experiment output folders

The large ConvNeXT checkpoint files are intentionally excluded from GitHub because they are larger than GitHub's normal 100 MB file limit.

### 3. CvT Experiments

`CvT vs ConvNeXT/cvt/`

This folder contains CvT experiment outputs and saved model checkpoints that are small enough to store in the repository. It includes several training variants, such as:

- CvT13 compile run
- CvT13 batch-size 64 run
- CvT13 SDPA/math optimized run
- CvT13 Food-101 model

The accompanying JSON files store validation and test metrics for these model runs.

### 4. Figures and Visualizations

`CvT vs ConvNeXT/figures/`

This folder contains plots and image outputs used to explain the experiments. The visualizations include:

- Dataset sample montages
- Training dashboards
- Accuracy and loss curves
- Speed and accuracy trade-off charts
- VRAM telemetry
- Dataset completeness summaries
- Model comparison charts
- Optimized iNaturalist sample predictions

### 5. Report

`Report.pdf`

This is the final written report for the Deep Learning project. It summarizes the motivation, model choices, experimental workflow, results, and conclusions.

## Technologies Used

- Python
- PyTorch
- TorchVision
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Deep learning image classification workflows
- CvT architecture
- ConvNeXT / ConvNeXT V2 architecture

## Model Checkpoints

Some local checkpoint files are not included in this GitHub repository because they are too large for normal GitHub storage:

- `CvT vs ConvNeXT/convnext/best.pt`
- `CvT vs ConvNeXT/convnext/last.pt`
- `CvT vs ConvNeXT/convnext/best_convnextv2_food101.pth`
- `CvT vs ConvNeXT/convnext/last_checkpoint.pth`
- `CvT vs ConvNeXT/convnext/convnextv2_tiny_food101_10epoch_b16/*.pth`
- `CvT vs ConvNeXT/convnext/smoke_outputs/*.pth`
- `CvT vs ConvNeXT/convnext/smoke_outputs_2/*.pth`

The repository still includes the experiment code, metrics, plots, report, and smaller CvT checkpoints that are suitable for GitHub.

## How to Use This Repository

1. Read `Report.pdf` for the project explanation and results.
2. Open `CvT vs ConvNeXT/all_model_training_testing_optimization.ipynb` to inspect the main training and evaluation workflow.
3. Review the metric files in `convnext/` and `cvt/` to compare model performance.
4. View the charts in `figures/` to understand training behavior, accuracy trends, and model trade-offs.
5. Use `CvT vs ConvNeXT/convnext/requirements.txt` as the starting point for recreating the ConvNeXT environment.

## Project Purpose

This project was created for a Deep Learning course. It demonstrates how two different deep learning model families can be trained, evaluated, and compared for image classification. The work emphasizes practical model experimentation, visual result interpretation, and evidence-based comparison between transformer-inspired and convolution-based architectures.

