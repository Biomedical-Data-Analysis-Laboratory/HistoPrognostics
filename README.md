# HistoBCG

This is the source code described in the paper "Self-Contrastive Weakly Supervised Learning Framework for Prognostic Prediction Using Whole Slide Images" by Saul Fuster, Farbod Khoraminia, Julio Silva-Rodríguez, Umay Kiraz, Geert J. L. H. van Leenders, Trygve Eftestøl, Valery Naranjo, Emiel A.M. Janssen, Tahlita C.M. Zuiverloon, and Kjersti Engan  - under revision.

### 1 - Abstract
We present a pioneering investigation into the application of deep learning techniques to analyze histopathological images for addressing the substantial challenge of automated prognostic prediction. Prognostic prediction poses a unique challenge as the ground truth labels are inherently weak, and the model must anticipate future events that are not directly observable in the image. To address this challenge, we propose a novel three-part framework comprising of a convolutional network based tissue segmentation algorithm for region of interest delineation, a contrastive learning module for feature extraction, and a nested multiple instance learning classification module. Our study explores the significance of various regions of interest within the histopathological slides and exploits diverse learning scenarios. The pipeline is initially validated on artificially generated data and a simpler diagnostic task. Transitioning to prognostic prediction, tasks become more challenging. Employing bladder cancer as use case, our best models yield an AUC of 0.721 and 0.678 for recurrence and treatment outcome prediction respectively.

<p align="center">
    <img src="images/pipeline overview.png">
</p>

### 2 - How to use

This codebase implements a two-part framework for image analysis: a contrastive learning module for feature extraction and a nested Multiple Instance Learning (MIL) classification module. The system offers the flexibility to use feature embeddings alone, clinicopathological data, or a combination of both for image classification tasks.

**Contrastive Learning Module:**
- The `main_XXX.py` scripts contain the implementation of a contrastive learning algorithm for training a convolutional neural network (CNN) backbone. Use `main_monoscale.py` and `main_multiscale.py` for single or tri-scale magnification inputs, respectively.
- The network is trained to generate feature embeddings for input images by maximizing the similarity between positive pairs and minimizing the similarity between negative pairs.
- You can configure the network architecture, loss function, and training hyperparameters in the script.
- Pre-trained models can be saved and loaded for feature extraction. Also, feature embeddings are generated for posterior classification training.

**Classification Module:**
- The `main_nmil.py` script builds a nested MIL classification model.
- It uses feature embeddings from the contrastive learning module, clinicopathological data, or both for image classification.
- The nested MIL approach hierarchically combines information from image regions (bags) to make a final classification decision.
- The script provides options for configuring the classifier architecture, handling multi-modal data, and specifying MIL pooling techniques.
- You can easily switch between using feature embeddings, clinicopathological data, or a combination as input.

**Usage:**
- Use the `main_XXX.py` script to train the contrastive learning module on your dataset. Provide data loaders and adjust training settings as needed.
- After feature extraction, you can use the `main_nmil.py` script to train the nested MIL classifier. Specify the input data sources and adjust the model architecture according to your requirements.
- The `inference.py` script allows you to perform inference on new images using the trained classification model.
- The `logistic_regression.py` script provides performance insights on the trained classification model.

**Dependencies:**
- Ensure that you have the required dependencies listed in the `requirements.txt` file.

<p align="center">
    <img src="images/heatmap.png">
</p>

### 3 - Link to paper
TBA

### 4 - How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper if you use it in your research.
```
TBA
```
