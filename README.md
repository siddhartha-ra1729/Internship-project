I am currently completing a 2-month online training and internship from Ardent, focused on Deep Learning using CNN and Keras. Throughout this internship, I have gained hands-on experience with cutting-edge technologies, including TensorFlow, while working on an eye-disease classification project. This project has sharpened my ability to build and optimize convolutional neural networks for real-world medical image classification tasks. Additionally, I am deepening my knowledge of essential deep learning concepts like transfer learning, data augmentation, and model evaluation. This internship has enhanced my technical skills in Python, machine learning algorithms, and the broader applications of artificial intelligence in solving complex problems.



# 👁️ Eye-Disease Classification using CNN and TensorFlow

This project focuses on developing an automated system for classifying various eye diseases using Convolutional Neural Networks (CNN) and Transfer Learning. It aims to support early detection and diagnosis of diseases such as **diabetic retinopathy**, **cataract**, and **glaucoma**.

---

## 🧠 Technologies Used
- Python 🐍
- TensorFlow & Keras
- CNN Architectures: VGG19, ResNet50, InceptionV3, MobileNetV2
- Data Augmentation & Image Preprocessing
- Visualization Libraries: Matplotlib, Seaborn, OpenCV, PIL

---

## 🎯 Objectives
- Accurately classify common eye diseases using CNN.
- Improve diagnostic efficiency with deep learning.
- Leverage transfer learning to enhance model accuracy.
- Develop an accessible solution for eye care diagnostics.

---

## 📁 Dataset
- **Source:** Kaggle
- **Content:** Retinal fundus images categorized by disease type.
- **Preprocessing:** Null value removal, normalization, augmentation, and label encoding.
- **Split:** Training and validation using K-Fold Cross Validation.

---

## 🏗️ Model Architectures Explored
- **VGG19:** Baseline CNN architecture with ~138M parameters.
- **ResNet50:** Deep residual network with skip connections.
- **InceptionV3:** Efficient feature extraction with low parameter count.
- **MobileNetV2:** Lightweight model for mobile deployment.
- **Custom CNN:** 2 Conv2D → MaxPooling2D → Flatten → Dense layers.

---

## 📊 Performance
- **Accuracy:**  
  - 84% with traditional CNN  
  - **94% with Transfer Learning**
- **Best Classifier Accuracy:**  
  - *99.89%* for glaucoma & diabetic retinopathy

- **Metrics Used:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  
  - ROC Curves

---

## 📈 Visualizations
- Original vs Augmented Images
- Class Distribution (Bar Chart)
- Accuracy vs Epochs (Line/Scatter Plot)
- Heatmaps & Boxplots of Model Performance
- Confusion Matrix for class-wise performance

---

## 🔍 Evaluation
- Early Stopping & Model Checkpoints used
- High variance between training and validation results in some cases
- Identified overfitting in base models, mitigated via augmentation and transfer learning

---

## 🚀 Future Scope
- Integrate Vision Transformers for advanced detection
- Add more diverse and rare eye disease datasets
- Develop mobile and lightweight model deployment
- Improve interpretability for clinical use

---
