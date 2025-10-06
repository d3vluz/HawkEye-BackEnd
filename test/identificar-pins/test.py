import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]

image_bgr = cv2.imread('../imagens_pins/pushpins_logical_anomalies_021.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# plt.imshow(image_rgb)
# plt.title('Imagem Original (RGB)')
# plt.axis('off')
# plt.show()

# Segmentação por Cor
hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([10, 165, 100])
upper_yellow = np.array([30, 255, 255])
mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# plt.imshow(mask, cmap='gray')
# plt.title('Máscara de Amarelo')
# plt.axis('off')
# plt.show()

# Detecção de Contornos

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 2000

filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        filtered_contours.append(contour)

image_with_contours = image_rgb.copy()

cv2.drawContours(image_with_contours, filtered_contours, -1, (255, 0, 255), 3)
num_objects = len(filtered_contours)

# plt.imshow(image_with_contours)
# plt.title(f'Contornos Filtrados ({num_objects} objetos)')
# plt.axis('off')
# plt.show()