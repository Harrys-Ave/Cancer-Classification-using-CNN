# Lung and Colon Cancer Classification Using Convolutional Neural Networks (CNNs)

**Author**: Harry Averkiadis  
**Institution**: MSc in Data Science  
**Date**: March 2024

This repository contains the code and analysis for a deep learning project aimed at multi-class classification of histopathological images for identifying types of lung and colon cancers. The primary goal was to evaluate the effectiveness of Convolutional Neural Networks (CNNs), including transfer learning with VGG16, in classifying medical images into five categories.

## Project Overview

The project focused on developing, improving, and evaluating CNN models for classifying lung and colon tissue images into the following categories:
1. Lung Benign Tissue
2. Lung Adenocarcinoma
3. Lung Squamous Cell Carcinoma
4. Colon Adenocarcinoma
5. Colon Benign Tissue

The primary goals were:
- To establish a baseline CNN model.
- To enhance the model with fine-tuning, regularization, and architectural changes.
- To apply transfer learning using a pre-trained VGG16 model on ImageNet for improved performance.

## Files Overview

- **Assignment_DL_Harry_Averkiadis.ipynb**: Contains the code for data preprocessing, model training, evaluation, and experimentation.
- **Deep Learning Report_Harry_Averkiadis.pdf**: Detailed report documenting the problem, methodology, results, and discussion of findings.

## Methodology

### 1. Data Preprocessing and Exploratory Data Analysis
- **Data Source**: LC25000 dataset with 25,000 histopathological images.
- **Resizing**: Images were resized to 120x120 pixels for consistency and computational efficiency.
- **Encoding and Splitting**: Labels were one-hot encoded, and data was split into training (60%), validation (20%), and test (20%) sets.
- **Class Distribution**: Ensured balanced class distribution across splits to prevent bias in the model.

### 2. Baseline Model
A simple CNN architecture was implemented as a baseline to assess initial model performance. The model showed moderate accuracy and precision, but signs of overfitting were evident after the 3rd epoch.

### 3. Improved Model with Fine-Tuning
- **Regularization**: L2 regularization and dropout layers were added to mitigate overfitting.
- **Learning Rate and Optimizer**: Tuned the learning rate of the Adam optimizer to 0.0001 for smoother convergence.
- **Architectural Changes**: Added convolutional layers and adopted a pyramid-like layer structure to capture more complex patterns.

### 4. Transfer Learning Model
- **VGG16**: Integrated a VGG16 pre-trained model with frozen layers to utilize its learned features.
- **Dense Layers**: Added custom fully connected layers for classification.
- **Early Stopping**: Implemented early stopping with a patience of 3 to avoid unnecessary epochs.

### 5. Evaluation Metrics
The models were evaluated using:
- **Accuracy, Precision, Recall, and F1 Score**
- **ROC Curves and Confusion Matrices** for each class to gauge model performance on unseen data.

## Results

- **Enhanced CNN Model**: Achieved a test accuracy of 98%, a notable improvement over the baseline accuracy of 78%.
- **Transfer Learning (VGG16)**: Reached a test accuracy of 97% with an AUC score of 1.00 for all classes.
  
The enhanced CNN model provided the best results, with significant improvements in accuracy and reduced false positives, crucial for medical diagnostics.

## Dependencies

- Python 3.8
- TensorFlow 2.4
- Keras 2.4
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Please refer to the notebook for the detailed code, architecture, and results.
