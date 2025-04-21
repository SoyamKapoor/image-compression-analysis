# 🖼️ Image Compression using JPEG and Wavelet Techniques

This project performs **Image Compression** using two prominent methods: **JPEG (DCT-based)** and **Wavelet Transform (JPEG2000-like)**. It evaluates quality using **PSNR**, **SSIM**, and **compression ratio**, with visual comparisons to highlight performance trade-offs.

---

## 📌 Project Overview

### Objectives:
- Load and compress images using JPEG (DCT) and Wavelet techniques
- Reconstruct images from compressed data
- Evaluate image quality using PSNR and SSIM
- Calculate actual compression ratio from file sizes
- Visualize original vs. reconstructed images

---

## 🧰 Tools & Libraries Used

- Python 3.x
- OpenCV (for image handling)
- PyWavelets (for wavelet compression)
- NumPy
- Matplotlib (for visualization)
- scikit-image (for PSNR & SSIM evaluation)

---

## 🧪 Key Steps in the Compression Process

1. **Image Loading**
   - Reads input image and detects color mode

2. **JPEG Compression**
   - Uses OpenCV's JPEG encoder with configurable quality
   - Decodes the image back to original dimensions

3. **Wavelet Compression**
   - Applies multilevel wavelet decomposition (`bior3.3`)
   - Reconstructs image using inverse wavelet transform

4. **Evaluation Metrics**
   - Calculates **PSNR** (Peak Signal-to-Noise Ratio)
   - Calculates **SSIM** (Structural Similarity Index)
   - Computes **compression ratio** based on file sizes

5. **Visualization**
   - Displays side-by-side comparison of original vs compressed images

---

## 🚀 How to Run the Script

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/image-compression-analysis.git
   cd image-compression-analysis

     ```

2. **Install Required Packages**
   Make sure you have Python 3.12.8 and Jupyter Notebook installed. You can install the required libraries using pip:

   ```bash
    pip install numpy opencv-python pywavelets matplotlib scikit-image

   ```

3. **Run the script**
   ```bash
   script.py
   ```

   ---

## 📌 Conclusion

This project demonstrates a comparative analysis of **JPEG** and **Wavelet** image compression methods.  
By evaluating **PSNR**, **SSIM**, and **compression ratio**, it highlights the trade-offs between **visual quality** and **file size**, helping guide the choice of technique in real-world applications like **storage**, **transmission**, and **multimedia systems**.
