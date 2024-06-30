import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# 导入skimage的库函数实现，用于对比手工实现的结果图像
from skimage.feature import hog    # 从skimage库中导入HOG特征计算函数
from skimage import data, io, color, exposure   # 导入示例图片和图像处理工具


class Hog_descriptor():

    #   初始化
    #   cell_size每个cell单元的像素数
    #   bin_size表示把360分为多少边

    def __init__(self, img, cell_size=256, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    #   获取hog向量和图片

    def extract(self):
        # 获得原图的shape
        height, width = self.img.shape
        # 计算原图的梯度大小
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        # cell_gradient_vector用来保存每个细胞的梯度向量
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        height_cell,width_cell,_ = np.shape(cell_gradient_vector)

        #   计算每个细胞的梯度直方图

        for i in range(height_cell):
            for j in range(width_cell):
                # 获取这个细胞内的梯度大小
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                # 获得这个细胞内的角度大小
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # 转化为梯度直方图格式
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # hog图像
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        # block为2x2
        for i in range(height_cell - 1):
            for j in range(width_cell - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    #   计算原图的梯度大小
    #   角度大小

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    #   分解角度信息到
    #   不同角度的直方图上

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    #   计算每个像素点所属的角度

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    #   将梯度直方图进行绘图

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

# 库函数实现
image_path = "photo2.png"
img_std = io.imread(image_path)
fd, hog_image = hog(img_std, orientations=8, pixels_per_cell=(50, 50),
                   cells_per_block=(1, 1), visualize=True, channel_axis=-1)

# 手工实现
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
hog = Hog_descriptor(img, cell_size=50, bin_size=8)
vector, my_hog_img = hog.extract()

# 使用matplotlib创建一个包含3个子图的画布
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
ax1.axis('off')
ax1.imshow(img_std)
ax1.set_title('Input image')

# 对HOG图像进行强度重缩放，以便更好地可视化
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.jet)
ax2.set_title('Histogram of Oriented Gradients')

my_hog_img_rescaled = exposure.rescale_intensity(my_hog_img, in_range=(0, 10))

ax3.axis('off')
ax3.imshow(my_hog_img_rescaled, cmap=plt.cm.jet)
ax3.set_title('My Histogram of Oriented Gradients')

plt.show()

