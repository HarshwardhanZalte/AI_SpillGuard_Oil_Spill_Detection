# Project Report: SpillGuard - AI-Driven Oil Spill Identification System


## 1. Summary

Oil spills represent one of the most severe threats to marine ecosystems, coastal economies, and global environmental health. Traditional detection methods, such as manual satellite imagery inspection or aerial surveillance, are often labor-intensive, time-consuming, and prone to human error due to "look-alikes" (e.g., biological slicks, low-wind areas).

This project, SpillGuard, introduces an automated, high-precision solution using Deep Learning. By leveraging a ResNet50-UNet architecture, the system performs Semantic Segmentation on satellite imagery (SAR and Optical) to identify oil spills at a pixel level. The project involved a complete lifecycle development: from data preprocessing and augmentation to model training with custom loss functions (Focal + Dice) to handle class imbalance. The final model achieved a Dice Coefficient of 92%, significantly outperforming baseline methods. The system was successfully deployed as a user-friendly Streamlit interface, allowing for real-time analysis, visualization, and reporting.

---

## 2. Table of Contents

1. [Introduction](#3-introduction)
2. [Dataset & Data Analysis](#4-dataset--data-analysis)
3. [Data Preprocessing (Milestone 1)](#5-data-preprocessing-milestone-1)
4. [Model Architecture (Milestone 2)](#6-model-architecture-milestone-2)
5. [Training Strategy & Loss Functions (Milestone 3)](#7-training-strategy--loss-functions-milestone-3)
6. [Experimental Results](#8-experimental-results)
7. [Deployment & Interface (Milestone 4)](#9-deployment--interface-milestone-4)
8. [Weekly Activity Log](#10-weekly-activity-log)
9. [Conclusion](#11-conclusion)

---

## 3. Introduction

### 3.1 Problem Statement

The ocean surface is vast, and oil spills can occur anywhere. In Synthetic Aperture Radar (SAR) imagery, oil spills appear as dark patches because oil dampens the capillary waves on the water surface, reducing radar backscatter. However, not all dark patches are oil. Natural phenomena like algae blooms, rain cells, and low-wind zones look identical to oil spills. Distinguishing "true oil" from these "look-alikes" is the primary challenge.

### 3.2 Objectives

- **Pixel-Level Precision:** Implement Semantic Segmentation rather than simple classification. We need to know the exact shape of the spill to estimate the volume of dispersants required.
- **Automation:** Remove the human bottleneck in satellite image analysis.
- **Robustness:** Achieve >90% accuracy despite the heavy class imbalance (95% water vs. 5% oil).

---

## 4. Dataset & Data Analysis

### 4.1 Dataset Structure

The dataset is organized into a directory structure suitable for supervised learning. It contains pairs of images: the raw satellite input and the corresponding "Ground Truth" mask.

- **Input Data (/image folder):** Contains SAR and Optical images (JPG/PNG). These images vary in resolution and lighting conditions.
- **Label Data (/mask folder):** Contains the segmentation masks. These are images where the pixel color indicates the class.
  - Black (0,0,0): Background (Water, Land, Ships)
  - Specific Color: Oil Spill

Dataset Link: [Download Dataset](https://zenodo.org/records/10555314)

### 4.2 Data Exploration

During the initial exploration phase, I noticed a significant class imbalance in the dataset. The majority of the images depicted vast stretches of open ocean, with actual oil slicks occupying only a small fraction of the total area. This observation was critical, as it led to the decision to incorporate Data Augmentation techniques to artificially expand the dataset and the use of Weighted Loss Functions to prevent the model from becoming biased toward the background class.

---

## 5. Data Preprocessing (Milestone 1)

Data quality is the foundation of high-performance AI. Our preprocessing pipeline focused on standardization and noise reduction.

### 5.1 Image Resizing Strategy

*Topic presented during Milestone 1 Review*

**The Challenge:** Satellite images come in varying resolutions (e.g., 1024x1024, 512x512). CNNs require fixed input dimensions.

**The Solution:** We standardized all inputs to 256x256 pixels.

**Code Implementation:**

```python
def resize_image(image, width=256, height=256, is_mask=False):
    """Resizes image to target dimensions."""
    # Crucial Distinction:
    # Use Nearest Neighbor for masks to preserve exact class values (0 or 1)
    # Use Bilinear/Linear for images to preserve smooth gradients
    method = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=method)
```

**Why this matters:** If we used `INTER_LINEAR` on a binary mask, the boundary between 0 (Black) and 1 (White) would become 0.5 (Gray). This "Gray" value confuses the model because it doesn't belong to any class. `INTER_NEAREST` ensures strict class boundaries.

### 5.2 Denoising (Speckle Noise)

SAR images suffer from "speckle noise" (granular interference). We applied a Median Blur filter, which is effective at removing salt-and-pepper noise while preserving edges.

```python
def reduce_speckle_noise(image):
    """Applies Median Blur to remove SAR speckle noise."""
    return cv2.medianBlur(image, 5)
```

### 5.3 Normalization

Neural networks converge faster when input values are small and centered. We normalized pixel intensity from the range [0, 255] to [0.0, 1.0].

---

## 6. Model Architecture (Milestone 2)

We moved beyond the standard U-Net to a Transfer Learning approach using ResNet50-UNet.

### 6.1 The Backbone: ResNet50

Instead of training the "Encoder" (feature extractor) from scratch, we used ResNet50 pre-trained on ImageNet.

- **Benefit:** The model starts with a learned understanding of edges, textures, and shapes.
- **Depth:** 50 layers allow for the extraction of deep semantic features that a shallow custom network would miss.

### 6.2 The Decoder & Skip Connections

The Decoder upsamples the features back to the original image size (256x256). We utilized Skip Connections to concatenate features from the ResNet encoder directly to the decoder.

**Code Snippet (Model Definition):**

```python
def build_resnet50_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    
    # Encoder: ResNet50 (Pre-trained)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Skip Connections from specific ResNet layers
    s1 = base_model.get_layer("conv1_relu").output          # 128x128
    s2 = base_model.get_layer("conv2_block3_out").output    # 64x64
    s3 = base_model.get_layer("conv3_block4_out").output    # 32x32
    s4 = base_model.get_layer("conv4_block6_out").output    # 16x16
    
    # Bridge
    bridge = base_model.get_layer("conv5_block3_out").output # 8x8

    # Decoder Block Example
    d1 = UpSampling2D((2, 2))(bridge)
    d1 = Concatenate()([d1, s4]) # Skip Connection
    d1 = Conv2D(512, (3, 3), padding="same", activation="relu")(d1)
    
    # ... (Subsequent decoder blocks) ...
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return Model(inputs, outputs)
```

---

## 7. Training Strategy & Loss Functions (Milestone 3)

### 7.1 The Class Imbalance Problem

*Topic presented during Milestone 3 Review*

In our dataset, oil spills often occupy only 1-5% of the image pixels. A standard model using Binary Cross Entropy (BCE) would achieve 95% accuracy by simply predicting "All Water." To fix this, we implemented a custom loss function.

### 7.2 Solution: Focal Loss + Dice Loss

We combined two powerful loss functions to force the model to learn the minority class.

- **Focal Loss:** Down-weights "easy" examples (water) and focuses on "hard" examples (oil).
- **Dice Loss:** Optimizes for the shape overlap (Intersection over Union).

**Code Implementation:**

```python
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        focal_term = (1 - p_t) ** gamma
        return tf.reduce_mean(focal_term * bce)
    return loss

def combined_loss(y_true, y_pred):
    # Combine Pixel-wise focus (Focal) with Shape focus (Dice)
    return focal_loss()(y_true, y_pred) + dice_loss(y_true, y_pred)
```

Download Trained Model: [Download Model](https://drive.google.com/file/d/1ODqgK7i7iKfzbSY3h50JYi15U6zfErfQ/view?usp=sharing)

---

## 8. Experimental Results

After training for 60 epochs with an Adam optimizer (Learning Rate 1e-4) and Early Stopping:

- **Training Accuracy:** 98.2%
- **Validation Dice Coefficient:** 0.920
- **Validation IoU:** 0.86

The high Dice score (0.92) indicates that the model is not just detecting the presence of oil, but accurately contouring its boundaries. The Confusion Matrix analysis revealed a very low False Negative rate, which is critical for environmental monitoring (we cannot afford to miss a spill).

---

## 9. Deployment & Interface (Milestone 4)

We deployed the trained model using Streamlit, creating a professional dashboard for end-users.

### 9.1 Post-Processing (Noise Removal)

Even with a good model, raw predictions often contain tiny "specks" of noise. In `app.py`, we implemented Morphological Opening to clean the masks before display.

**Code from app.py:**

```python
# 4. Noise Removal (Morphological Opening)
if NOISE_SIZE > 1:
    kernel = np.ones((NOISE_SIZE, NOISE_SIZE), np.uint8)
    # Erosion followed by Dilation removes small white specks
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
```

### 9.2 Visualization Features

The app provides three distinct views for analysis:

- **Overlay:** A semi-transparent Red mask blended onto the original image.
- **Heatmap:** A "Plasma" colormap showing the model's confidence probability per pixel.
- **Analytics:** A histogram of pixel probabilities to assess model certainty.

**Code for Overlay:**

```python
if np.any(mask_indices):
    # Blend Original Image (70%) with Red Mask (30%)
    overlay[mask_indices] = (original_size_img[mask_indices] * (1-ALPHA) + 
                           mask_colored[mask_indices] * ALPHA).astype(np.uint8)
```

---

## 10. Weekly Activity Log

| Week | Phase | Detailed Activities & Accomplishments |
|------|-------|--------------------------------------|
| **Week 1** | Data Collection | • Sourced SAR/Optical dataset<br>• Cleaned corrupt files<br>• Split data into Train (80%), Val (10%), Test (10%) |
| **Week 2** | Preprocessing | • **Milestone 1 Presentation:** "Why and How We Resize Images" (Bilinear vs Nearest Neighbor)<br>• Coded resizing and denoising pipeline<br>• Implemented Median Blur for SAR speckle noise |
| **Week 3** | Model Design | • Studied Semantic Segmentation architectures<br>• Implemented baseline U-Net in TensorFlow<br>• Designed Input/Output layers (256x256x3 -> 256x256x1) |
| **Week 4** | Optimization | • Identified accuracy plateau with standard U-Net<br>• Researched Transfer Learning<br>• Implemented ResNet50 backbone integration<br>• Matched Encoder/Decoder dimensions |
| **Week 5** | Training Setup | • **Milestone 3 Presentation:** "Loss Functions and Accuracy Metrics" (Focal + Dice Loss)<br>• Coded custom loss functions<br>• Configured callbacks: ModelCheckpoint, EarlyStopping |
| **Week 6** | Training | • Executed full training loops on GPU<br>• Tuned hyperparameters (Batch Size: 16, LR: 1e-4)<br>• Achieved 92% Validation Accuracy<br>• Saved best model weights (.h5) |
| **Week 7** | Visualization | • Created Python scripts for side-by-side comparison<br>• Developed "Overlay" blending function<br>• Generated Confusion Matrices and IoU charts |
| **Week 8** | Deployment | • Built frontend using Streamlit<br>• Integrated model and preprocessing logic<br>• Added heatmaps, histograms, and report download features<br>• Finalized documentation |

---

## 11. Conclusion

The SpillGuard project successfully demonstrated the efficacy of Deep Learning in automating environmental protection tasks. By combining a robust ResNet50-UNet architecture with advanced loss functions (Focal + Dice), we achieved a high-precision model capable of segmenting oil spills with 92% accuracy.

### Key Technical Achievements

- **Effective Preprocessing:** Handling multi-resolution satellite data without losing label integrity
- **Advanced Architecture:** Utilizing Transfer Learning to overcome the limitations of shallow networks
- **Production-Ready Deployment:** Creating a polished Streamlit interface with built-in noise removal and reporting tools

### Future Work

Future work will focus on integrating real-time drone video feeds and expanding the class definitions to detect oil thickness levels, further aiding cleanup efforts.