import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


input_folder = r""
output_img_folder = r""
output_hist_folder = r""

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_hist_folder, exist_ok=True)


sigma = 15

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f'图像读取失败: {img_path}')
            continue


        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy_img = img.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)


        noisy_img_name = f'gauss_{os.path.splitext(filename)[0]}.png'
        noisy_img_path = os.path.join(output_img_folder, noisy_img_name)

        result, encoded_img = cv2.imencode('.png', noisy_img)
        if result:
            with open(noisy_img_path, mode='wb') as f:
                encoded_img.tofile(f)
            print(f'成功保存高斯噪声图像: {noisy_img_path}')
        else:
            print(f'编码失败: {noisy_img_path}')
            continue


        plt.figure(figsize=(10, 5))

        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.hist(img.ravel(), bins=256, color='gray')
        plt.title('Original Histogram')

        plt.subplot(2, 2, 3)
        plt.imshow(noisy_img, cmap='gray')
        plt.title(f'Gaussian Noise (σ={sigma})')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.hist(noisy_img.ravel(), bins=256, color='gray')
        plt.title('Noisy Histogram')

        plt.tight_layout()

        hist_img_name = f'gauss_hist_{os.path.splitext(filename)[0]}.png'
        hist_img_path = os.path.join(output_hist_folder, hist_img_name)
        plt.savefig(hist_img_path)
        plt.close()

print("所有图像高斯噪声处理完成")
