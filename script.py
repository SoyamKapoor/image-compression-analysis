import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class ImageCompressor:
    def __init__(self):
        pass

    def compress_dct(self, img, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return encimg

    def decompress_dct(self, compressed, is_color):
        img_array = np.frombuffer(compressed, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE)

    def compress_jpeg2000(self, img, compression_ratio=20):
        encode_param = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio]
        result, encimg = cv2.imencode('.jp2', img, encode_param)
        return encimg

    def decompress_jpeg2000(self, compressed, is_color):
        img_array = np.frombuffer(compressed, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE)

    def evaluate(self, original, reconstructed):
        is_color = len(original.shape) == 3
        if is_color:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            reconstructed_gray = reconstructed
        return psnr(original_gray, reconstructed_gray), ssim(original_gray, reconstructed_gray)

    def calculate_actual_compression(self, original_path, compressed_path):
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        return original_size / compressed_size if compressed_size != 0 else 0

    def plot_images(self, original, jpeg_img, jp2_img, is_color, save_path):
        if is_color:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            jpeg_img = cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB)
            jp2_img = cv2.cvtColor(jp2_img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 5))
        titles = ['Original', 'JPEG Compressed', 'JPEG2000 Compressed']
        images = [original, jpeg_img, jp2_img]

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(images[i], cmap=None if is_color else 'gray')
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def plot_file_size_comparison(self, sizes_kb, save_path):
        labels = list(sizes_kb.keys())
        sizes = list(sizes_kb.values())

        plt.figure(figsize=(7, 5))
        plt.bar(labels, sizes, color=['gray', 'blue', 'green'])
        plt.ylabel('File Size (KB)')
        plt.title('File Size Comparison')
        plt.grid(axis='y')
        plt.savefig(save_path)
        plt.show()

    def plot_pixels_comparison(self, images, save_path):
        labels = ['Original', 'JPEG', 'JPEG2000']
        pixels = [img.shape[0] * img.shape[1] for img in images]

        plt.figure(figsize=(7, 5))
        plt.bar(labels, pixels, color=['gray', 'blue', 'green'])
        plt.ylabel('Number of Pixels (Width × Height)')
        plt.title('Pixel Count Comparison')
        plt.grid(axis='y')
        plt.savefig(save_path)
        plt.show()

    def run_experiment(self, img_path, jpeg_quality=75, jp2_compression=20):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        is_color = len(img.shape) == 3
        original_path = Path(img_path)

        output_dir = Path("compression_results")
        output_dir.mkdir(exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        compressed_jpeg = self.compress_dct(img, jpeg_quality)
        reconstructed_jpeg = self.decompress_dct(compressed_jpeg, is_color)
        jpeg_path = output_dir / "jpeg_compressed.jpg"
        cv2.imwrite(str(jpeg_path), reconstructed_jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        jpeg_ratio = self.calculate_actual_compression(str(original_path), str(jpeg_path))
        psnr_jpeg, ssim_jpeg = self.evaluate(img, reconstructed_jpeg)

        compressed_jp2 = self.compress_jpeg2000(img, jp2_compression)
        reconstructed_jp2 = self.decompress_jpeg2000(compressed_jp2, is_color)
        jp2_path = output_dir / "jpeg2000_compressed.jp2"
        cv2.imwrite(str(jp2_path), reconstructed_jp2)

        jp2_as_jpg_path = output_dir / "jpeg2000_converted_to_jpeg.jpg"
        cv2.imwrite(str(jp2_as_jpg_path), reconstructed_jp2)

        jp2_ratio = self.calculate_actual_compression(str(original_path), str(jp2_path))
        psnr_jp2, ssim_jp2 = self.evaluate(img, reconstructed_jp2)

        print("\n=== Compression Results ===")
        print(f"Original Image Size: {os.path.getsize(original_path)/1024:.2f} KB")
        print(f"JPEG Quality: {jpeg_quality}")
        print(f"JPEG2000 Compression Parameter: {jp2_compression}")

        print("\nJPEG Compression:")
        print(f"  File Size: {os.path.getsize(jpeg_path)/1024:.2f} KB")
        print(f"  Compression Ratio: {jpeg_ratio:.2f}:1")
        print(f"  PSNR: {psnr_jpeg:.2f} dB, SSIM: {ssim_jpeg:.4f}")

        print("\nJPEG2000 Compression:")
        print(f"  File Size: {os.path.getsize(jp2_path)/1024:.2f} KB")
        print(f"  Compression Ratio: {jp2_ratio:.2f}:1")
        print(f"  PSNR: {psnr_jp2:.2f} dB, SSIM: {ssim_jp2:.4f}")

        self.plot_images(
            img, reconstructed_jpeg, reconstructed_jp2, is_color,
            save_path=figures_dir / "comparison_images.png"
        )

        sizes_kb = {
            "Original": os.path.getsize(original_path)/1024,
            "JPEG": os.path.getsize(jpeg_path)/1024,
            "JPEG2000": os.path.getsize(jp2_path)/1024,
        }
        self.plot_file_size_comparison(
            sizes_kb,
            save_path=figures_dir / "file_size_comparison.png"
        )

        self.plot_pixels_comparison(
            [img, reconstructed_jpeg, reconstructed_jp2],
            save_path=figures_dir / "pixel_count_comparison.png"
        )

        return {
            'original': {'path': original_path},
            'jpeg': {'path': jpeg_path, 'psnr': psnr_jpeg, 'ssim': ssim_jpeg, 'ratio': jpeg_ratio},
            'jpeg2000': {
                'path': jp2_path,
                'converted_jpg': jp2_as_jpg_path,
                'psnr': psnr_jp2,
                'ssim': ssim_jp2,
                'ratio': jp2_ratio
            }
        }

if __name__ == "__main__":
    compressor = ImageCompressor()
    results = compressor.run_experiment(
        img_path="E:/VS_CODE/python_codes/image_compression_for_efficient_storage_and_transmission/colorful_image.png",
        jpeg_quality=80,
        jp2_compression=20
    )