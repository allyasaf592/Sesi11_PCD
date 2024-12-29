import imageio as img
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def sobel_edge_detection(image):
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    edge_magnitude = np.hypot(dx, dy)
    edge_magnitude = edge_magnitude / np.max(edge_magnitude) * 255
    return edge_magnitude.astype(np.uint8)

def local_thres(image, block_size=15, c=10):
    imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    threshold = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_area = imgPad[i:i+block_size, j:j+block_size]
            local_mean = np.mean(local_area)
            threshold[i, j] = 255 if image[i, j] > (local_mean - c) else 0

    return threshold

def basicThres(image, level):
    threshold = np.where(image > level, 255, 0)
    return threshold.astype(np.uint8)

original_image = img.imread("E:/Data Alya/Sem 5/PengolahanCitraDigital/singa.jpg", mode='F')

sobel_image = sobel_edge_detection(original_image)

local_thres_image = local_thres(sobel_image)

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Deteksi Tepi (Sobel)")
plt.imshow(sobel_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Segmentasi Citra (Thresholding Lokal)")
plt.imshow(local_thres_image, cmap='gray')
plt.axis('off')

plt.show()
