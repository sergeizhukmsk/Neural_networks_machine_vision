import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Просмотр содержимого папки Image
print(os.listdir('./image/'))


# Загрузка изображения
image = cv2.imread('./image/portret_50.jpg', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('./image/portret_75.jpg', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('./image/nerpa.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка, удалось ли загрузить изображение
if image is None:
    print("Ошибка: изображение не загружено. Проверьте путь к файлу.")
else:
    print("Изображение успешно загружено.")

# Применение оператора Собеля для вычисления градиентов по осям X и Y
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Градиент по оси X
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Градиент по оси Y

# Вычисление величины градиента
gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

# Нормализация градиентов для отображения
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# Отображение результатов
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('Оригинальное изображение')
plt.imshow(image, cmap='gray')

plt.subplot(1, 4, 2)
plt.title('Градиент по оси X')
plt.imshow(sobel_x, cmap='gray')

plt.subplot(1, 4, 3)
plt.title('Градиент по оси Y')
plt.imshow(sobel_y, cmap='gray')

plt.subplot(1, 4, 4)
plt.title('Величина градиента')
plt.imshow(gradient_magnitude, cmap='gray')

plt.show()
