# FWMNet: Wavelet Transform Attention Network for THz Imaging Enhancement
Official implementation of the paper:  
**"Wavelet Transform Attention Network for Low-Exposure THz Image Enhancement in Rydberg Atomic Systems"**

> **Abstract**: This work proposes a novel Full-Wavelet Attention Network (FWMNet) to address the challenges of low exposure and high noise in Rydberg atom THz imaging systems. By integrating discrete wavelet transform with channel-spatial attention mechanisms, our method achieves state-of-the-art performance in THz image reconstruction.

## 🚀 Quick Start
### Propare model and datasets 
- Pretrained model:[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/robojun/FWMNet)
- Datasets:[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/datasets/robojun/THz_imaging)
### Installation
```bash
# Create conda environment
conda create -n fwmnet python=3.10

# Activate environment
conda activate fwmnet

# Install dependencies
pip install -r requirements.txt

# train
python train.py

# eval
python evaluate_fwmnet.py
```
# Research Overview
<img src="https://raw.githubusercontent.com/ssrfwanjun/FWMNet/main/overview.png" width="100%">
## 🧪 Background and Motivation

Terahertz (THz) radiation (0.1-10 THz) has emerged as a powerful modality for security screening and biomedical imaging due to its unique properties: non-ionizing nature, material penetration capability, and molecular fingerprinting. However, traditional THz imaging systems face significant limitations in achieving both high speed and high quality simultaneously.

Rydberg atom-based THz detection represents a breakthrough technology that leverages quantum effects to convert invisible THz field information into visible fluorescence signals through atomic energy level transitions. This "THz-to-optical" conversion mechanism bypasses the traditional trade-offs in direct THz detection, enabling high-resolution, high-frame-rate imaging at room temperature.

## 🔬 Problem Statement

Despite the theoretical advantages of Rydberg atom detection, practical high-speed imaging (e.g., 100 fps) introduces critical challenges:

- **Severe underexposure**: Millisecond-scale exposure times drastically reduce signal photon counts
- **Low signal-to-noise ratio (SNR)**: Readout noise and fluorescence background degrade image quality
- **Detail loss**: Fine structures and textures are overwhelmed by noise
- **Dynamic imaging limitations**: Quantitative analysis of rapid processes becomes challenging

These limitations significantly impact the practical application of Rydberg atom THz imaging in real-world scenarios such as security inspection and biomedical diagnostics.

## 💡 Innovative Solution: FWMNet Architecture

We present a novel **Full-Wavelet Attention Network (FWMNet)** that synergistically integrates wavelet transform with attention mechanisms for high-fidelity THz image restoration.

### Core Technical Components

| Component | Function | Innovation Value |
|----------|----------|------------------|
| **Discrete Wavelet Transform (DWT)** | Decomposes images into multi-scale frequency subbands (LL, LH, HL, HH) | Natural separation of signal and noise with physical priors |
| **Dual Attention Mechanism** | Adaptive feature weighting through channel and spatial attention | Intelligent feature selection and noise suppression |
| **Selective Kernel Feature Fusion** | Dynamic multi-scale feature aggregation | Preserves both global structure and local textures |
| **Recursive Residual Design** | Progressive signal decomposition | Enables deep network construction with stable training |

[](@replace=1)


## 🚀 Technical Advantages and Performance

### Quantitative Superiority
- **PSNR improvement**: +7.23 dB (from 18.01 dB to 29.75 dB)
- **SSIM enhancement**: +8.91% (from 0.76 to 0.93)
- **Training efficiency**: Achieves excellent performance with only 300 image pairs
- **Generalization capability**: Robust performance across different imaging scenarios

### Qualitative Excellence
Compared to traditional methods (histogram equalization, Gaussian filtering) and prior deep learning approaches, FWMNet demonstrates superior detail preservation and artifact suppression while maintaining natural visual quality.

[](@replace=2)


## 🌊 Dynamic Imaging Applications

The method successfully demonstrates real-time capability in capturing dynamic processes, particularly in fluid interface analysis:

### Water-Ethanol Flow Imaging
- **Clear boundary visualization**: Distinct phase separation between immiscible liquids
- **Interface oscillation analysis**: Quantified spatiotemporal dynamics at 10.1 Hz fundamental frequency
- **Amplitude measurement**: 0.7 mm oscillation amplitude revealing viscous-capillary instabilities

[](@replace=3)


## 📊 Experimental Validation

### Dataset Construction
Carefully curated paired dataset acquired through controlled acquisition protocol:
- **300 image pairs** (270 training, 30 testing)
- **Precise alignment**: Same scene captured at different frame rates (100+ fps vs 10 fps)
- **Diverse samples**: Metal targets, optical elements, fluidic devices
- **Quality assurance**: Strict spatial alignment and content verification

### Ablation Studies
Systematic evaluation confirms optimal configuration:
- **64 convolutional filters** with **4-level depth** provides best performance-complexity tradeoff
- **Dual attention mechanism** significantly improves feature selection
- **Multi-scale feature fusion** enhances both local and global information integration

[](@replace=4)


## 🎯 Key Innovations

1. **First application of wavelet-attention networks** to Rydberg atom THz high-speed imaging
2. **Unpaired training strategy** enabling practical deployment with limited data
3. **Spatial-frequency cooperative learning** paradigm for optimal detail preservation
4. **Real-time dynamic imaging capability** validated through fluid flow experiments

## 🔮 Future Directions

- **Optimization of inference speed** for real-time processing applications
- **Extension to biological tissue imaging** and other emerging THz applications
- **Integration with computational imaging** techniques for further performance enhancement
- **Exploration of multi-modal sensing** combining THz with other imaging modalities

This work establishes a new benchmark for THz image enhancement and demonstrates the potential of deep learning-powered computational imaging for advancing quantum sensing technologies.
