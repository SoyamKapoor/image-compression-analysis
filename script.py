import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ImageCompressor:
    def __init__(self):
        pass

    def compress_dct(self, img, quality):
        """Compress image using OpenCV's JPEG encoder."""
        is_color = len(img.shape) == 3
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return encimg

    def decompress_dct(self, compressed, quality, is_color):
        """Decompress JPEG-compressed image."""
        img_array = np.frombuffer(compressed, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE)

    def compress_wavelet(self, img, wavelet_name='bior3.3', level=3):
        """Compress using wavelet transform."""
        coeffs = []
        is_color = len(img.shape) == 3
        if is_color:
            for c in cv2.split(img):
                coeff = pywt.wavedec2(c, wavelet=wavelet_name, level=level)
                coeffs.append(coeff)
        else:
            coeffs = pywt.wavedec2(img, wavelet=wavelet_name, level=level)
        return coeffs

    def decompress_wavelet(self, coeffs, wavelet_name='bior3.3', is_color=True):
        """Reconstruct image from wavelet coefficients."""
        if is_color:
            channels = []
            for coeff in coeffs:
                rec = pywt.waverec2(coeff, wavelet=wavelet_name)
                rec = np.clip(rec, 0, 255).astype(np.uint8)
                channels.append(rec)
            return cv2.merge(channels)
        else:
            rec = pywt.waverec2(coeffs, wavelet=wavelet_name)
            return np.clip(rec, 0, 255).astype(np.uint8)

    def evaluate(self, original, reconstructed):
        """Calculate PSNR and SSIM."""
        is_color = len(original.shape) == 3
        if is_color:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        return psnr(original, reconstructed), ssim(original, reconstructed)

    def calculate_actual_compression(self, original_path, compressed_path):
        """Calculate compression ratio using file sizes."""
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        return original_size / compressed_size if compressed_size != 0 else 0

    def visualize(self, original, compressed, title, is_color):
        if is_color:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap=None if is_color else 'gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(compressed, cmap=None if is_color else 'gray')
        plt.title(title)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def run_experiment(self, img_path, jpeg_quality=75, wavelet_level=3, wavelet_name='bior3.3'):
        """Run JPEG and Wavelet compression, visualize and evaluate."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        is_color = len(img.shape) == 3
        output_dir = Path("compression_results")
        output_dir.mkdir(exist_ok=True)

        # --- JPEG Compression ---
        compressed_dct = self.compress_dct(img, jpeg_quality)
        reconstructed_dct = self.decompress_dct(compressed_dct, jpeg_quality, is_color)
        psnr_dct, ssim_dct = self.evaluate(img, reconstructed_dct)

        jpeg_path = output_dir / "jpeg_compressed.jpg"
        cv2.imwrite(str(jpeg_path), reconstructed_dct, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        jpeg_ratio = self.calculate_actual_compression(img_path, jpeg_path)

        # --- Wavelet Compression ---
        coeffs_wavelet = self.compress_wavelet(img, wavelet_name, wavelet_level)
        reconstructed_wavelet = self.decompress_wavelet(coeffs_wavelet, wavelet_name, is_color)
        psnr_wavelet, ssim_wavelet = self.evaluate(img, reconstructed_wavelet)

        wavelet_path = output_dir / "wavelet_compressed.jpg"
        cv2.imwrite(str(wavelet_path), reconstructed_wavelet, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        wavelet_ratio = self.calculate_actual_compression(img_path, wavelet_path)

        # --- Print Results ---
        print("\n=== Compression Results ===")
        print(f"Original Image Size: {os.path.getsize(img_path)/1024:.2f} KB")
        print(f"JPEG Quality: {jpeg_quality}")
        print(f"Wavelet: {wavelet_name}, Levels: {wavelet_level}")

        print("\nJPEG Compression:")
        print(f"  File Size: {os.path.getsize(jpeg_path)/1024:.2f} KB")
        print(f"  Compression Ratio: {jpeg_ratio:.2f}:1")
        print(f"  PSNR: {psnr_dct:.2f} dB, SSIM: {ssim_dct:.4f}")

        print("\nWavelet Compression:")
        print(f"  File Size: {os.path.getsize(wavelet_path)/1024:.2f} KB")
        print(f"  Compression Ratio: {wavelet_ratio:.2f}:1")
        print(f"  PSNR: {psnr_wavelet:.2f} dB, SSIM: {ssim_wavelet:.4f}")

        # --- Visualize ---
        self.visualize(img, reconstructed_dct, f"JPEG Compressed (Q={jpeg_quality})", is_color)
        self.visualize(img, reconstructed_wavelet, f"Wavelet Compressed ({wavelet_name})", is_color)

        return {
            'jpeg': {'path': jpeg_path, 'psnr': psnr_dct, 'ssim': ssim_dct, 'ratio': jpeg_ratio},
            'wavelet': {'path': wavelet_path, 'psnr': psnr_wavelet, 'ssim': ssim_wavelet, 'ratio': wavelet_ratio}
        }


# === Run the experiment ===
if __name__ == "__main__":
    compressor = ImageCompressor()
    results = compressor.run_experiment(
        img_path="colorful_image.png",
        jpeg_quality=80,
        wavelet_level=4,
        wavelet_name='bior3.3'
    )