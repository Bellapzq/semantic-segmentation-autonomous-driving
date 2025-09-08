# WildScenes Semantic Segmentation

This repository contains our group project **"Vision Pro: WildScenes Image Segmentation"** for **COMP9517: Computer Vision (UNSW, Term 2 2024)**.  
The goal of the project was to **develop and evaluate deep learning methods for semantic segmentation of natural environments**, using the [WildScenes dataset](https://csiro-robotics.github.io/WildScenes/).

---

## 📖 Project Overview
Autonomous vehicles must navigate safely in complex, noisy natural environments.  
Unlike urban scenes, forests and natural landscapes pose unique challenges due to irregular objects, overlapping structures, and class imbalance.  

In this project, we explored **semantic segmentation** of forest imagery to classify each pixel into one of 15 semantic classes (e.g., foliage, trunk, water, mud, sky).  

- **Dataset**: WildScenes Benchmark (9306 annotated images, 2016×1512 resolution)  
- **Task**: Fine-grained semantic segmentation of natural scenes  
- **Evaluation metric**: Mean Intersection-over-Union (mIoU)  

---

## ⚙️ Methods Implemented

### 1. U-Net + Superpixels Preprocessing
- Used **U-Net** architecture with a **ConvNeXt encoder**.  
- Integrated **SLIC superpixel preprocessing** to enhance boundary detection.  
- Loss: CrossEntropy | Optimizer: Adam | Scheduler: Step decay.  

### 2. Ensemble Model (UNet++ + PSPNet)
- Combined **UNet++** (for fine-grained segmentation) with **PSPNet** (for global context).  
- Encoders: ResNet-50 pretrained on ImageNet.  
- Loss: CrossEntropy | Optimizer: Adam | Scheduler: StepLR.  

### 3. DeepLabV3 + Dual Encoders
- Implemented **DeepLabV3 with Atrous Spatial Pyramid Pooling (ASPP)**.  
- Extended with **dual encoders (ResNet-101 + EfficientNet-B3)** to capture both complex and lightweight features.  
- Loss: Focal loss (to handle class imbalance).  

---

## 🧪 Experiments
- **Data split**: followed official WildScenes train/val/test split with 6052 train, 2134 test, 284 validation.  
- **Due to hardware limits**: trained on subset of 1400 train, 300 test, 300 validation images.  
- **Preprocessing**: images resized to 512×512, with augmentation (contrast, brightness, blur).  

---

## 📊 Results

| Method                     | mIoU  |
|-----------------------------|-------|
| U-Net + Superpixels         | 0.15  |
| Ensemble (UNet++ + PSPNet)  | 0.39  |
| DeepLabV3 + Dual Encoders   | 0.42  |

- **Ensemble model** improved class-wise segmentation (e.g., Bush, Dirt, Sky, Tree-foliage).  
- **DeepLabV3 + Dual Encoders** performed best overall, especially for challenging classes (Log, Water, Rock), but still below WildScenes benchmarks.  
- Results highlight the difficulty of segmentation in noisy natural images with limited training data.  

---

## 📺 Deliverables
- [Report (PDF)](./VisionPro9517FinalReport.pdf)  
- [Presentation video ](https://drive.google.com/file/d/1LkP_g7P-GkiijGlseAyj-h8Y8daxdQnT/view?usp=drive_link) (Google cloud)
- Source code (this repo)  

---

## 👩‍💻 Contributions
- **Bella Pang**: Model implementation (U-Net + preprocessing, DeepLabV3 dual encoders), training/validation experiments, result visualization, and report writing.  
- Team members: Literature review, baseline coding, PSPNet/UNet++ ensemble, presentation.  

---

## 📚 References
- Vidanapathirana, K. et al. (2023). *WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Large-scale Natural Environments.* [arXiv:2312.15364](https://arxiv.org/abs/2312.15364)  
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
- Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). *Pyramid Scene Parsing Network (PSPNet).* CVPR.  
- Chen, L.C., Papandreou, G., Schroff, F., & Adam, H. (2017). *Rethinking Atrous Convolution for Semantic Segmentation.*  

---