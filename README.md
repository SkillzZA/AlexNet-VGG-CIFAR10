


# Comparative Study: AlexNet vs. VGG on CIFAR-10

This repository contains a Jupyter/Colab notebook that implements and compares adapted versions of the AlexNet and VGG convolutional neural network architectures for image classification on the CIFAR-10 dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)

## Overview

The primary goal of this project is to:
1.  Describe and compare the design philosophies of AlexNet and VGG architectures.
2.  Implement adapted versions suitable for the 32x32 image size of CIFAR-10 using PyTorch.
3.  Train these models (AlexNet, VGG-16, VGG-16 with Batch Normalization, VGG-8) on CIFAR-10.
4.  Analyze and compare their performance based on accuracy, loss curves, training time, and overfitting behavior.
5.  Visualize learned filters and feature maps from early layers.
6.  Evaluate the impact of adding Batch Normalization to VGG-16.
7.  Evaluate the effect of reducing network depth (VGG-8 vs VGG-16).

## Dataset

*   **CIFAR-10:** A standard benchmark dataset consisting of 60,000 32x32 color images in 10 classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks), with 50,000 training images and 10,000 test images.

## Architectures Implemented

*   AlexNet (adapted for CIFAR-10)
*   VGG-16 (adapted for CIFAR-10, *initially required Kaiming initialization to train effectively*)
*   VGG-16 with Batch Normalization (adapted for CIFAR-10)
*   VGG-8 (shallower VGG variant, adapted for CIFAR-10)

## File

*   `AlexNetVGG.ipynb`: The main Jupyter/Colab notebook containing all code, analysis, visualizations, and results.

## Requirements

*   Python 3
*   PyTorch
*   Torchvision
*   NumPy
*   Matplotlib

## Usage Instructions

1.  **Clone the Repository or Download the Notebook:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```
    Or download the `.ipynb` file directly
    
    **Or JUST PRESS THE OPEN IN COLAB BUTTON 😡** 
    
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)<br/> 


   $~$


 &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)



  &nbsp; &nbsp; &nbsp; &nbsp;   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)   &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)
  
  
  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkillzZA/AlexNet-VGG-CIFAR10/blob/main/AlexNetVGG.ipynb)
  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; 



  $~$
  
3.  **Open in Google Colab:**
    *   Go to [Google Colab](https://colab.research.google.com/).
    *   Select `File -> Upload notebook...` and upload the `.ipynb` file.
4.  **Set Runtime:**
    *   In Colab, go to `Runtime -> Change runtime type`.
    *   Select `GPU` as the hardware accelerator. This is crucial for reasonable training times.
5.  **Run the Cells:**
    *   Execute the cells sequentially from top to bottom (`Shift + Enter` or the play button).
    *   **Note:** The training cells for the models will take a significant amount of time to run (potentially 10-30+ minutes per model depending on Colab's allocated resources and the number of epochs).
6.  **View Results:**
    *   The notebook contains embedded outputs, including training logs (loss/accuracy per epoch, time), plots of the training curves, and visualizations of filters/feature maps.
    *   Markdown cells within the notebook provide analysis and discussion based on the generated results.

## Results Summary

The notebook trains and evaluates the different architectures. Key findings discussed within the notebook include:
*   Baseline comparison of adapted AlexNet and VGG-16.
*   The significant performance improvement and faster convergence provided by Batch Normalization (VGG-16-BN vs VGG-16).
*   The performance trade-off related to network depth on CIFAR-10 (VGG-8 vs VGG-16).
*   The necessity of Kaiming initialization for training the plain VGG-16 effectively.

All detailed results, plots, and discussions can be found within the `AlexNetVGG.ipynb` notebook file.
