# 🧩 Diffusion Model with Transformer for Image Denoising

---

## 📘 Overview
This project implements a **simplified Diffusion Model** integrated with a **Transformer-based U-Net** for **image denoising** tasks.  
The system was trained on the **MNIST dataset** to reconstruct clean images from noisy inputs using a hybrid architecture that blends **CNN feature extraction** and **Transformer-based global attention**.

---

## ⚙️ Architecture

### 1. Denoising Model (U-Net)
- Encoder-decoder CNN inspired by U-Net.  
- Encoder: multiple convolutional layers + ReLU + max pooling.  
- Decoder: transposed convolutions for upsampling.

### 2. Transformer Integration
- Multi-head attention (4 heads).  
- Feedforward network (128 hidden units).  
- Layer normalization for stability.  
- Captures long-range dependencies to improve denoising quality.

---

## 🌫️ Diffusion Process
- Simplified Gaussian noise simulation.  
- Single-step denoising operation without time-conditioned noise levels.  
- Faster training, focused on evaluating Transformer performance.

---

## 🧮 Training & Evaluation
**Dataset:** MNIST (60k training / 10k testing)  
**Noise level:** σ = 0.3  
**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam (lr = 0.001)  
**Epochs:** 50  

### Evaluation Metrics:
- **MSE:** 0.0045  
- **SSIM:** 0.9142  

---

## 📊 Results
The Transformer-enhanced U-Net demonstrated superior reconstruction quality, producing visually clear denoised images even under moderate noise conditions.

| Metric | Score |
|---------|-------|
| MSE     | 0.0045 |
| SSIM    | 0.9142 |

Sample visualization:
- Original MNIST images  
- Noisy images (Gaussian noise)  
- Denoised outputs (model predictions)

---

## 🔍 Attention Map
Due to limited compute resources, the attention visualization was generated for a single Transformer block (head 1).  
Even at this scale, the model displayed meaningful attention patterns focused on key image structures.

---

## 🧠 Key Learnings
- Integrating attention mechanisms enhances global context awareness in image denoising.  
- Simplified diffusion approaches can deliver solid results with constrained computational budgets.  
- The synergy between **U-Net** and **Transformer** architectures boosts robustness against noise.

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- NumPy / Matplotlib  
- SSIM & MSE evaluation metrics  

---

## 🔗 Future Work
- Extend to multi-step DDPM framework.  
- Apply to higher-resolution datasets (CIFAR-10, CelebA).  
- Optimize attention layer for better interpretability.

