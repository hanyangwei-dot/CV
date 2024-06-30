# 导入必要的库
import numpy as np                  
import matplotlib.pyplot as plt     
from skimage.feature import hog    # 从skimage库中导入HOG特征计算函数
from skimage import data, io, color, exposure   # 导入示例图片和图像处理工具


# 计算HOG特征
# 参数解释：
# - image: 输入图像
# - orientations: 方向bins的数量，默认为9，这里设置为8
# - pixels_per_cell: 每个cell的像素大小，设置为(16, 16)
# - cells_per_block: block中cell的数量，设置为(1, 1)，意味着没有进一步聚合
# - visualize: 是否生成可视化的HOG图像，默认为False，这里设为True
# - multichannel: 如果输入是RGB图像，是否单独处理每个通道，默认为False，这里设为True
# img = data.astronaut()
# print(img.shape)

image_path = "photo1.png"
image = io.imread(image_path)
# image_gray = color.rgb2gray(img)  # 转为灰度图像
# print(image_gray)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualize=True, channel_axis=-1)
# 返回hog特征向量和hog可视化图像

# 使用matplotlib创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  

# 第一个子图：原始图像

# imshow显示图像，cmap=plt.cm.gray使用灰度颜色映射
# set_title设置子图标题
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# 对HOG图像进行强度重缩放，以便更好地可视化
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 第二个子图：HOG特征的可视化图像
# 同样关闭坐标轴，显示重缩放后的HOG图像，并设置标题
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.jet)
ax2.set_title('Histogram of Oriented Gradients')

# 显示画布上的所有子图
plt.show()