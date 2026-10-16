import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_folder = r""
output_img_folder = r""
output_hist_folder = r""

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_hist_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f'读取失败: {img_path}')
            continue

        equalized_img = cv2.equalizeHist(img)


        eq_img_path = os.path.join(output_img_folder, f'eq_{os.path.splitext(filename)[0]}.png')
        success = cv2.imwrite(eq_img_path, equalized_img)
        if not success:
            print(f'写入失败: {eq_img_path}')
        else:
            print(f'写入成功: {eq_img_path}')


        plt.figure(figsize=(10, 5))
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.hist(img.ravel(), bins=256, color='gray')
        plt.title('Original Histogram')

        plt.subplot(2, 2, 3)
        plt.imshow(equalized_img, cmap='gray')
        plt.title('Equalized Image')
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.hist(equalized_img.ravel(), bins=256, color='gray')
        plt.title('Equalized Histogram')
        plt.tight_layout()

        hist_img_path = os.path.join(output_hist_folder, f'hist_{os.path.splitext(filename)[0]}.png')
        plt.savefig(hist_img_path)
        plt.close()

print('所有图片处理完成')
