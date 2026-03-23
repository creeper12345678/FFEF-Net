# 🌍 FFEF-Net: Frequency‑Guided Feature Enhancement and Fusion Network  
### 🛰️ High‑Resolution Settlement Mapping in West Africa from SDGSAT‑1 Multispectral Imagery

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2403.12345-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
[![GitHub stars](https://img.shields.io/github/stars/nkszjx/TreeSegment_SA?style=social)](https://github.com/nkszjx/TreeSegment_SA)

---

## 🔍 Overview

**FFEF-Net** is a deep learning framework designed for precise **settlement segmentation** from SDGSAT‑1 multispectral imagery. It addresses two critical challenges:

- **Spectral interference** (noise, band gaps) in SDGSAT‑1 data  
- **Multi‑scale feature fusion** for complex settlement morphologies

The model introduces two novel modules:

- **CSFE** (Cross‑Channel Spectral Feature Extraction) – decouples spatial and spectral information to enhance discriminability.
- **FFEF** (Frequency‑Guided Feature Enhancement and Fusion) – adaptively fuses low‑frequency semantics and high‑frequency details.

The method achieves **79.49% IoU** and **88.57% F1‑score** on a diverse West African test set, outperforming UNet, DeepLabV3+, and other state‑of‑the‑art networks.

> **📦 Associated Product:** The resulting 10 m settlement maps for four West African regions (Kebbi, Nasarawa, Sassandra‑Marahoué, Mamou) are freely available via [Zenodo](#).

---

## ✨ Key Features

- ✅ **First settlement mapping study** using SDGSAT‑1 multispectral data in West Africa  
- ✅ **Frequency‑guided fusion** – preserves both semantic integrity and boundary detail  
- ✅ **Robust to sensor noise** – CSFE suppresses strip noise and radiometric anomalies  
- ✅ **Scalable architecture** – can be integrated with other backbones  
- ✅ **State‑of‑the‑art performance** – IoU +9.6% over previous specialised networks  

---

## 🧠 Architecture

![FFEF-Net architecture](docs/architecture.png)  
*Fig. 1: FFEF‑Net framework. The encoder includes a CSFE block followed by a ConvNeXt backbone; the decoder uses PPM and three FFEF modules for progressive multi‑scale fusion.*

For details, see our paper: [link to preprint](#).

---

## 📂 Repository Structure
