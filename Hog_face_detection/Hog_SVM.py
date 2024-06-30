import cv2 as  cv
from sklearn.datasets import fetch_lfw_people  # 人脸数据集
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import PatchExtractor

'''
获取标注好的人脸数据集，LFW (Labeled Faces in the Wild) 
数据集可从http://vis-www.cs.umass.edu/lfw/获取）。
'''

faces = fetch_lfw_people(download_if_missing=False)

positive_patches = faces.images
print(positive_patches.shape)

from skimage import data, transform, color

imgs_to_use = ['hubble_deep_field', 'text', 'coins', 'moon',
               'page', 'clock', 'coffee', 'chelsea', 'horse']

images = []
for name in imgs_to_use:

    if len(np.array(getattr(data, name)()).shape) == 3:
        images.append(color.rgb2gray(getattr(data, name)()))
    else:
        images.append(getattr(data, name)())

print(len(images))

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extract_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extract_patch_size,
                               max_patches=N, random_state=0)

    patches = extractor.transform(img[np.newaxis])
    # patches = extractor.transform(img)

    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])

    return patches


negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])


print(negative_patches.shape)

# SVM进行模型训练
from skimage import feature
from itertools import chain

X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

grad = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)

grad.fit(X_train, y_train)

# print(grad.best_score_)
# print(grad.best_params_)

model = grad.best_estimator_
model.fit(X_train, y_train)

# 新图像测试
test_img = cv.imread('scientists.jpg')
test_img = color.rgb2gray(test_img)
test_img = transform.rescale(test_img, 0.5)
# test_img = test_img[:120, 60:140]   # 选取位置

plt.imshow(test_img, cmap='gray')
plt.axis('off')
plt.show()


# 使用滑动窗口，找出合格的窗口
def sliding_window(img, patch_size=positive_patches[0].shape,
                   istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0]- Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i: i + Ni, j: j+Nj]

            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


# 使用滑动窗口计算每一视窗的HOG
indices, patches = zip(*sliding_window(test_img))
patches_hog = np.array([feature.hog(patch) for patch in patches])

labels = model.predict(patches_hog)
print(labels.sum())  # 检测到的窗口总数 55

fig, ax = plt.subplots()
ax.imshow(test_img, cmap='gray')
ax.axis('off')

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='green',
                               alpha=0.3, lw=2, facecolor='none'))

plt.show()


# NMS
# 剔除多余的窗口
def non_max_suppression_slow(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = max(x2[i], x2[j])
            yy2 = max(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick]


condidate_box = []
for i, j in indices[labels == 1]:
    condidate_box.append([j, i, Nj, Ni])

final_boxes = non_max_suppression_slow(np.array(condidate_box).reshape(-1, 4))

fig, ax = plt.subplots()

ax.imshow(test_img, cmap='gray')
ax.axis('off')

for i, j, Ni, Nj in final_boxes:
    ax.add_patch(plt.Rectangle((i, j), Ni, Nj, edgecolor='green',
                               alpha=0.3, lw=2, facecolor='none'))

plt.show()
