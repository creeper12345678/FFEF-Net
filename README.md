# 🌍 FFEF-Net: Frequency‑Guided Feature Enhancement and Fusion Network  
### 🛰️ High‑Resolution Settlement Mapping in West Africa from SDGSAT‑1 Multispectral Imagery

---

## 🔍 Overview

**FFEF-Net** is a deep learning framework designed for precise **settlement segmentation** from SDGSAT‑1 multispectral imagery. 

The model introduces two novel modules:

- **CSFE** (Cross‑Channel Spectral Feature Extraction) – decouples spatial and spectral information to enhance discriminability.
- **FFEF** (Frequency‑Guided Feature Enhancement and Fusion) – enhances low‑frequency semantics and high‑frequency details.

The method achieves **79.49% IoU** and **88.57% F1‑score** on a West African test set, outperforming UNet, DeepLabV3+, and other state‑of‑the‑art networks.

## 📦 Associated Product
Our training and test datasets and the 10 m settlement products for four West African regions (Kebbi, Nasarawa, Sassandra‑Marahoué, Mamou) are freely available via [Figshare](https://doi.org/10.6084/m9.figshare.31829476).


## 🚀 Getting Started

Our codebase is built upon MMsegmentation

