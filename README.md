# Image Cartoonization and Custom Filter Application

## Overview

This repository presents a comprehensive pipeline for image cartoonization and custom filter application, leveraging state-of-the-art techniques in computer vision and deep learning. The goal of this project is to transform real-life photographs into cartoon-style images, followed by the application of custom filters (such as glasses, masks, and hats) using facial landmarks. The process involves data preprocessing, training of Convolutional Neural Networks (CNN), CNN with Neural Architecture Search (NAS), and Vision Transformers (ViT), cartoonization using OpenCV and Pillow, and the application of custom filters based on both trained models and Haar Cascade classifiers.

## Table of Contents

1. [Project Description](#project-description)
2. [Setup Instructions](#setup-instructions)
3. [Pipeline Overview](#pipeline-overview)
4. [Key Features](#key-features)
5. [Model Training Details](#model-training-details)
6. [Cartoonization Process](#cartoonization-process)
7. [Facial Landmark-Based Filter Application](#facial-landmark-based-filter-application)
8. [Comparison Results](#comparison-results)
9. [Acknowledgements](#acknowledgements)
10. [License](#license)

## Project Description

This project provides a method for transforming facial images into cartoon-style representations while preserving their artistic essence. The cartoonization process involves simplifying edges, amplifying colors, and reducing noise, followed by the application of stylized enhancements. The system then detects facial landmarks and applies custom filters such as glasses, masks, and hats precisely to these detected areas. Both deep learning-based and traditional methods (Haar Cascade Classifiers) for detecting facial landmarks are explored and compared.

The deep learning models trained in this repository include:
- **CNN** for basic feature extraction.
- **CNN with Neural Architecture Search (NAS)** for optimizing the CNN structure.
- **Vision Transformers (ViT)** for capturing long-range dependencies in images.

## Setup Instructions

### 1. Prerequisites

To run the code and replicate the experiments, ensure you have the following libraries installed:

- Python 3.6+
- TensorFlow 2.x (for model training and inference)
- OpenCV (for image processing)
- Pillow (for image manipulation and cartoonization)
- dlib or OpenCV (for facial landmark detection)
- NumPy (for matrix operations)
- Matplotlib (for visualization)



### 2. Data Preparation

The dataset used for training is a collection of facial images. You can use any suitable facial image dataset (such as CelebA or LFW), or you can collect your own images. Ensure that the images are preprocessed as described below before training.

### 3. Training the Models

To train the models, follow the steps below:

1. **CNN Model Training**: 
    - Train a CNN model on the preprocessed dataset to detect and extract features from facial images.
    - Run the `train_cnn.py` script to begin training the CNN model.

2. **CNN with NAS**:
    - Use Neural Architecture Search (NAS) to optimize the CNN architecture. Run the `train_cnn_nas.py` script for this step.

3. **Vision Transformers (ViT)**:
    - Train the Vision Transformer model on the same dataset. Execute the `train_vit.py` script.

After the training is complete, the models will be saved as `.h5` files for future use.

## Pipeline Overview

The pipeline is divided into the following major steps:

1. **Data Preprocessing**:
   - Images are resized, normalized, and augmented to increase variability in the dataset.
   - Facial landmarks are detected using pre-trained models and annotated for filter application.

2. **Model Training**:
   - The CNN, CNN with NAS, and ViT models are trained to detect features in facial images and generate stylized outputs.

3. **Cartoonization**:
   - OpenCV and Pillow libraries are used to cartoonize images by simplifying edges, amplifying colors, and reducing noise.
   - Artistic enhancements, such as texture overlays and color transformations, are applied to create a visually appealing cartoon effect.

4. **Facial Landmark-Based Filter Application**:
   - Custom filters (e.g., glasses, hats) are applied based on the facial landmarks detected by the trained model (`.h5` file).
   - Haar cascade classifiers are also used to detect facial features and apply filters for comparison.

5. **Comparison and Evaluation**:
   - The results of the Haar cascade filter application and the `.h5`-based model filter application are compared to evaluate accuracy, visual appeal, and computational efficiency.

## Key Features

- **Data Preprocessing**: Includes facial landmark detection and image normalization to ensure high-quality input data.
- **Model Training**: The models trained include CNN, CNN with NAS, and ViT for optimal performance.
- **Cartoonization**: Uses OpenCV and Pillow to create cartoon-style images with enhanced features.
- **Facial Landmark-Based Filter Application**: Allows the overlay of custom filters using both deep learning models and Haar Cascade classifiers.
- **Performance Comparison**: Compares the results of Haar cascade filters and deep learning-based filters to assess their strengths and weaknesses.

## Model Training Details

1. **CNN Model**:
   - A basic CNN is trained to extract features from facial images for cartoonization. The model captures spatial hierarchies in the images, learning to detect facial features and apply basic transformations.

2. **CNN with NAS**:
   - NAS is used to automatically search for the best architecture for the CNN model, optimizing layer structures and hyperparameters to enhance performance and reduce computational complexity.

3. **Vision Transformers (ViT)**:
   - ViTs are trained to understand long-range dependencies and contextual relationships within images, helping to refine the stylized cartoon features and enhance the global structure of the cartoonized images.

## Cartoonization Process

The cartoonization process involves two key libraries: **OpenCV** and **Pillow**.

- **OpenCV Filters**: Used for edge detection, color quantization, and noise reduction to create the initial cartoon effect (the "Cartoon Jack" effect).
- **Pillow Enhancements**: Applied for finer artistic adjustments such as edge enhancement, stylized color transformations, and texture overlays.

These techniques work together to create a cartoon-style image that is both visually appealing and artistically rich.

## Facial Landmark-Based Filter Application

The system applies custom filters based on facial landmark detection, performed using both the trained `.h5` model and Haar Cascade classifiers.

- **Custom Filters**: Filters such as glasses, hats, and masks are overlaid on the detected facial landmarks.
- **Haar Cascade Filters**: Haar cascades are used as an alternative method to detect facial features and apply corresponding filters.

The application of both techniques is compared in terms of accuracy, alignment, and computational performance.

## Comparison Results

After generating cartoonized images and applying custom filters, the results from both Haar Cascade and `.h5`-based methods are compared. Key evaluation metrics include:
- **Accuracy**: How well the filters are aligned with the facial landmarks.
- **Visual Quality**: A subjective measure of the overall appearance and appeal of the cartoonized images.
- **Computational Efficiency**: The time taken for filter application and model inference.

## Acknowledgements

- **OpenCV**: For providing powerful image processing tools used in cartoonization.
- **Pillow**: For enabling artistic enhancements on the cartoonized images.
- **dlib/OpenCV**: For providing pre-trained models for facial landmark detection.

