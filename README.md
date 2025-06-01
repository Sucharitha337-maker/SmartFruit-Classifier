# SmartFruit-Classifier

##  Project Overview

This project presents a machine learning-based approach for fruit classification, focusing on distinguishing **Golden Apples** and **Pears** using image data. It combines **Principal Component Analysis (PCA)** for dimensionality reduction and the **K-Nearest Neighbors (KNN)** algorithm for classification.

The project uses both RGB image analysis and segmented measurement data, leveraging tools such as **Python**, **ImageJ**, **JMP**, and **Photoshop**. Classification accuracy achieved up to **97.84%** with PCA (3 components) and KNN (K=2).

---

##  Dataset

- Source: [Fruits-360 Dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
- Classes Used: Golden Apple & Pear  
- Training Samples:  
  - Golden Apple: 480  
  - Pear: 492  
- Testing Samples:  
  - Golden Apple: 160  
  - Pear: 164  

---

##  Tools & Technologies

- **Python** – For PCA, KNN implementation, and visualization  
- **JMP** – PCA analysis and visualizations from measurements  
- **ImageJ** – Image segmentation and preprocessing  
- **Photoshop** – Manual image adjustments (if needed)

---

##  Implementation Workflow

###  Case Study 1: RGB Image Classification
1. **Preprocessing** – Resized all images to 100x100 pixels  
2. **PCA Transformation** – Reduced dimensions to 1, 2, and 3 components  
3. **KNN Classification** – Tested with various K values; selected K=2  
4. **Evaluation** – Achieved:
   - 2 PCA: 30.56% accuracy  
   - 3 PCA: **97.84% accuracy**  
   - Confusion Matrix plotted for evaluation  

###  Case Study 2: Segmented Image Classification
1. **Segmentation via ImageJ** – Converted RGB to binary black & white  
2. **Measurement Extraction** – Used `Analyze > Measure` to get features  
3. **PCA via JMP** – Analyzed variance, eigenvalues, and plotted components  
4. **KNN in Python** – Accuracy: **97.13%** using K=2  

---

##  Results Summary

| Method | PCA Components | K Value | Accuracy |
|--------|----------------|---------|----------|
| Case Study 1 (RGB) | 3 | 2 | **97.84%** |
| Case Study 2 (Segmented) | 3 | 2 | **97.13%** |

---

##  Conclusions

- PCA effectively reduced dimensionality while retaining key features  
- KNN proved to be a simple yet powerful classifier for visual fruit data  
- Image segmentation combined with measurement analysis yielded robust performance  

---

##  Future Work

- Expand dataset with more fruit types for multi-class classification  
- Apply advanced models like **Convolutional Neural Networks (CNN)** for better accuracy and generalization  
- Automate segmentation and measurement pipeline for larger-scale data processing

