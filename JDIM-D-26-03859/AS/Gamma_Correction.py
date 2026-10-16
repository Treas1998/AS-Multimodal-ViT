import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


input_folder = r""
output_img_folder = r""
output_hist_folder = r""

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_hist_folder, exist_ok=True)


gamma = 0.5
inv_gamma = 1.0 / gamma


gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")


for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f'图像读取失败: {img_path}')
            continue


        gamma_img = cv2.LUT(img, gamma_table)


        gamma_img_name = f'gamma_{os.path.splitext(filename)[0]}.png'
        gamma_img_path = os.path.join(output_img_folder, gamma_img_name)

        result, encoded_img = cv2.imencode('.png', gamma_img)
        if result:
            with open(gamma_img_path, mode='wb') as f:
                encoded_img.tofile(f)
            print(f'✅ 成功保存 Gamma 图像: {gamma_img_path}')
        else:
            print(f'❌ 编码失败: {gamma_img_path}')
            continue

        # 保存对比直方图
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.hist(img.ravel(), bins=256, color='gray')
        plt.title('Original Histogram')

        plt.subplot(2, 2, 3)
        plt.imshow(gamma_img, cmap='gray')
        plt.title(f'Gamma Corrected Image (γ={gamma})')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.hist(gamma_img.ravel(), bins=256, color='gray')
        plt.title('Gamma Histogram')

        plt.tight_layout()

        hist_img_name = f'gamma_hist_{os.path.splitext(filename)[0]}.png'
        hist_img_path = os.path.join(output_hist_folder, hist_img_name)
        plt.savefig(hist_img_path)
        plt.close()

print("所有图像Gamma校正处理完成！")
