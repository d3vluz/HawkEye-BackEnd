import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]

image_bgr = cv2.imread('C:/Users/usuario/Documents/Meus Programas/HawkEye-BackEnd/test/imagens_pins/pushpins_logical_anomalies_021.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([10, 165, 100])
upper_yellow = np.array([30, 255, 255])
mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

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

kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 5)
sure_bg = cv2.dilate(opening, kernel, iterations=1)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.22 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

image_for_watershed = image_rgb.copy()
markers = cv2.watershed(image_for_watershed, markers)

image_with_separated_contours = image_rgb.copy()
new_num_objects = 0

for label in np.unique(markers):
    if label <= 1:
        continue

    object_mask = np.zeros(mask.shape, dtype="uint8")
    object_mask[markers == label] = 255
    
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(image_with_separated_contours, [contour], -1, (255, 0, 255), 3)
            new_num_objects += 1


plt.imshow(image_with_separated_contours)
plt.title(f'7. Contornos Finais Separados ({new_num_objects} objetos)')
plt.axis('off')
plt.show()