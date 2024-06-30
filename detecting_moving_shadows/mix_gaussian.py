import multiprocessing as mp
import cv2
import numpy as np

'''
alpha 是论文中的更新速率 learning rate，alpha = 1 / default_history
default_history 表示训练得到背景模型所用到的集合大小，默认为 500，并且如果不手动设置 learning rate 的话，这个变量 default_history 就被用于计算当前的
learning rate，此时 default_history 越大，learning rate 就越小，背景更新也就越慢
'''

default_history = 200
# 方差阈值，用于判断当前像素是前景还是背景
default_var_threshold = 4.0 * 4.0
# 每个像素点高斯模型的最大个数，默认为 5
default_nmixtures = 5
default_background_ratio = 0.9
# 方差阈值，用于是否存在匹配的模型，如果不存在则新建一个
default_var_threshold_gen = 3.0 * 3.0
# 初始化一个高斯模型时，方差的值默认为 15
default_var_init = 15.0
# 高斯模型中方差的最大值为 5 * 15 = 75
default_var_max = 5 * default_var_init
# 高斯模型中方差的最小值为 4
default_var_min = 4.0
default_ct = 0.05

CV_CN_MAX = 512

FLT_EPSILON = 1.19209e-07


class GuassInvoker():
    def __init__(self, image, mask, gmm_model, mean_model, gauss_modes, nmixtures, lr, Tb, TB, Tg, var_init, var_min,
                 var_max, prune, ct, nchannels):
        self.image = image
        self.mask = mask
        self.gmm_model = gmm_model
        self.mean_model = mean_model
        self.gauss_modes = gauss_modes
        self.nmixtures = nmixtures
        self.lr = lr
        self.Tb = Tb
        self.TB = TB
        self.Tg = Tg
        self.var_init = var_init
        self.var_min = var_min
        self.var_max = var_max
        self.prune = prune
        self.ct = ct
        self.nchannels = nchannels

    # 针对原图像中的某一行进行处理，row 是图像中某一行的下标
    def calculate(self, row):
        lr = self.lr
        gmm_model = self.gmm_model
        mean_model = self.mean_model
        # image 中的某一行像素
        data = self.image[row]
        cols = data.shape[0]

        # 遍历原图像中的某一行的所有列
        for col in range(cols):
            background = False
            fits = False
            # 当前像素点的高斯模型个数
            modes_used = self.gauss_modes[row][col]
            total_weight = 0.

            # 当前像素点使用的所有高斯模型
            gmm_per_pixel = gmm_model[row][col]
            # 当前像素点使用的所有高斯模型的均值
            mean_per_pixel = mean_model[row][col]

            costs = []
            flag = False

            for mode in range(modes_used):
                # 一个 gmm 结构体的结构为: [weight, variance, c]，所以只有 c 值大于 0，才会进行计算 abs(x - mean) / variance
                # c 值表示和import os
                # from shutil import copy, rmtree
                # import random
                #
                #
                # def mk_file(file_path: str):
                #     if os.path.exists(file_path):
                #         # 如果文件夹存在，则先删除原文件夹在重新创建
                #         rmtree(file_path)
                #     os.makedirs(file_path)
                #
                #
                # def main():
                #     # 保证随机可复现
                #     random.seed(0)
                #
                #     # 将数据集中10%的数据划分到验证集中
                #     split_rate = 0.1
                #
                #     cwd = os.getcwd()
                #     data_root = os.path.join(cwd, "mushroom_data")
                #     origin_mushroom_path = os.path.join(data_root, "Mushrooms")
                #     assert os.path.exists(origin_mushroom_path), "path '{}' does not exist.".format(origin_mushroom_path)
                #
                #     mushroom_class = [cla for cla in os.listdir(origin_mushroom_path)
                #                     if os.path.isdir(os.path.join(origin_mushroom_path, cla))]
                #
                #     # 建立保存训练集的文件夹
                #     train_root = os.path.join(data_root, "train")
                #     mk_file(train_root)
                #     for cla in mushroom_class:
                #         # 建立每个类别对应的文件夹
                #         mk_file(os.path.join(train_root, cla))
                #
                #     # 建立保存验证集的文件夹
                #     val_root = os.path.join(data_root, "val")
                #     mk_file(val_root)
                #     for cla in mushroom_class:
                #         # 建立每个类别对应的文件夹
                #         mk_file(os.path.join(val_root, cla))
                #
                #     for cla in mushroom_class:
                #         cla_path = os.path.join(origin_mushroom_path, cla)
                #         images = os.listdir(cla_path)
                #         num = len(images)
                #         # 随机采样验证集的索引
                #         eval_index = random.sample(images, k=int(num*split_rate))
                #         for index, image in enumerate(images):
                #             if image in eval_index:
                #                 # 将分配至验证集中的文件复制到相应目录
                #                 image_path = os.path.join(cla_path, image)
                #                 new_path = os.path.join(val_root, cla)
                #                 copy(image_path, new_path)
                #             else:
                #                 # 将分配至训练集中的文件复制到相应目录
                #                 image_path = os.path.join(cla_path, image)
                #                 new_path = os.path.join(train_root, cla)
                #                 copy(image_path, new_path)
                #             print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
                #         print()
                #
                #     print("processing done!")
                #
                #
                # if __name__ == '__main__':
                #     main()对应高斯模型匹配的像素点的个数
                if gmm_per_pixel[mode][2] > 0:
                    _cost = np.sum((np.array(data[col]) - np.array(mean_per_pixel[mode])) ** 2)
                    _cost = np.sqrt(_cost / float(gmm_per_pixel[mode][1]))
                    # 将 [mode, abs(x - mean) / variance] 的值保存到 costs 中，其中 mode 是高斯模型的索引
                    costs.append(np.array([mode, _cost]))
                    flag = True

            min_cost = 4.0 * 4.0

            if flag:
                _, min_cost_index = np.argmin(np.array(costs), axis=0)
                min_index, min_cost = costs[min_cost_index]

            # 只有当前 abs(x - mean) / variance < Tg 时，才会认为像素点的值符合当前高斯模型
            if flag and min_cost < self.Tg:

                fits = True
                # 计算当前像素点所有高斯模型中 c 值之和，c 值也就是每一个高斯模型匹配上的像素点的个数，加一是因为现在有匹配上了一个像素点
                sum_c = np.sum(gmm_per_pixel[:, 2]) + 1
                # 计算权重 weight 的更新速率值，在之前的算法中，权重 weight，方差 variance 以及均值 mean 的更新速率全部为一个相同的固定值，
                # 而改进的算法中，weight, variance, mean 都是由不同速率进行更新，加快收敛速度
                lr_w = max(1.0 / sum_c, self.lr)

                # 遍历每一个高斯模型
                for mode in range(modes_used):
                    # gmm 高斯模型是一个 3 维向量，组成为 [weight, variance, c]
                    gmm = gmm_per_pixel[mode]
                    # 当前高斯模型匹配上的像素点的个数
                    c_mode = gmm[2]
                    # 对所有高斯模型的 weight 权重值进行更新，更新的公式为：weight = (1 - lr_w) * weight - lr_w * ct + lr_w * q
                    # 其中 q 只有对于当前最符合的高斯模型，值才为 1，其余的都为 0
                    weight = (1 - lr_w) * gmm[0] - lr_w * self.ct
                    mean = mean_per_pixel[mode]

                    var = gmm[1]
                    d_data = data[col] - mean
                    dist2 = np.sum(d_data ** 2)

                    # 使用马氏距离来判断当前像素点是属于前景还是背景
                    if total_weight < self.TB and dist2 < self.Tb * var:
                        background = True

                    swap_count = 0
                    # 如果当前高斯模型时最符合的
                    if mode == min_index:

                        gmm[2] += 1
                        # 当前高斯模型是最符合的，因此 q = 1，因此 weight 需要加上 lr_w * q = lr_w
                        weight += lr_w
                        # lr_mean 时均值 mean 的更新速率, lr_var 是方差 variance 的更新速率
                        lr_mean = max(1.0 / c_mode, self.lr)
                        lr_var = self.lr

                        if c_mode > 1:
                            lr_var = max(1.0 / (c_mode - 1), self.lr)

                        # 分别使用 lr_mean 和 lr_var 更新 mean 和 variance
                        mean = mean + lr_mean * d_data
                        mean_per_pixel[mode] = mean
                        var = var + lr_var * (dist2 - var)

                        var = max(var, self.var_min)
                        var = min(var, self.var_max)

                        gmm[1] = var

                        for i in range(mode, 0, -1):
                            if weight < gmm_per_pixel[i - 1][0]:
                                break
                            swap_count += 1
                            gmm_per_pixel[i - 1], gmm_per_pixel[i] = gmm_per_pixel[i], gmm_per_pixel[i - 1]
                            mean_per_pixel[i - 1], mean_per_pixel[i] = mean_per_pixel[i], mean_per_pixel[i - 1]

                    # 保证下一次模型的权重非负，lr_w * ct，也   就是 weight 要大于 lr_w * ct，否则当前高斯模型的 weight 可能就会出现小于 0 的情况。
                    # 如果 weight 小于常量值，那么必须将当前像素点的 nmodes 减一，也就是舍弃掉当前高斯模型，实现论文里面所说的模型个数自适应变化
                    if weight < self.ct * lr_w:
                        weight = 0
                        modes_used -= 1

                    gmm_per_pixel[mode - swap_count][0] = weight
                    total_weight += weight

            if not fits and lr > 0:
                # 如果模型数已经达到最大 nmixtures 时，替换掉权值最小的那个高斯模型; 否则，新增一个高斯模型
                if modes_used == self.nmixtures:
                    mode = self.nmixtures - 1
                else:
                    mode = modes_used
                    modes_used += 1

                # 如果只有一个高斯模型，那么就把这个高斯模型的权重设置为 1
                if modes_used == 1:
                    gmm_per_pixel[mode][0] = 1.0
                else:
                    # 新增加模型的权重等于 lr，也就是 learning rate
                    # 当前像素点有 nmixtures 个高斯模型，并且这些高斯模型是按照权重大小降序排列的
                    gmm_per_pixel[mode][0] = lr

                    for i in range(mode):
                        gmm_per_pixel[i][0] *= (1 - lr)

                # 初始化新的高斯模型的均值 mean，使用的就是原始图像中的像素点的值来进行初始化
                mean_per_pixel[mode] = data[col]
                # 初始化新增的混合高斯模型 gmm 的方差 variance
                gmm_per_pixel[mode][1] = self.var_init
                # 初始化新增高斯模型的 c 值为 1
                gmm_per_pixel[mode][2] = 1

                # 对所有的高斯模型按照权重进行降序排序
                for i in range(modes_used - 1, 0, -1):
                    if lr < gmm_per_pixel[i - 1][0]:
                        break
                    gmm_per_pixel[i - 1], gmm_per_pixel[i] = gmm_per_pixel[i], gmm_per_pixel[i - 1]
                    mean_per_pixel[i - 1], mean_per_pixel[i] = mean_per_pixel[i], mean_per_pixel[i - 1]

            self.gauss_modes[row][col] = modes_used
            self.mask[row][col] = 0 if background else 255

            # 对 gmm_per_pixel 中的各个高斯模型的权重值进行归一化
            total_weight = np.sum(gmm_per_pixel[:, 0])
            inv_factor = 0.
            if abs(total_weight) > FLT_EPSILON:
                inv_factor = 1.0 / total_weight

            gmm_per_pixel[:, 0] *= inv_factor

        return row, self.mask[row], self.gmm_model[row], self.mean_model[row], self.gauss_modes[row]


# noinspection PyAttributeOutsideInit
class GuassMixBackgroundSubtractor():
    def __init__(self):
        self.frame_count = 0
        self.history = default_history
        self.var_threshold = default_var_threshold
        self.nmixtures = default_nmixtures
        self.var_init = default_var_init
        self.var_max = default_var_max
        self.var_min = default_var_min
        self.var_threshold_gen = default_var_threshold_gen
        self.ct = default_ct
        self.background_ratio = default_background_ratio

    def apply(self, image, lr=-1):
        if self.frame_count == 0 or lr >= 1:
            self.initialize(image)

        self.image = image
        self.frame_count += 1
        # 计算 learning rate，也就是 lr，有以下三种情况：
        # 1.输入 lr 为 -1，那么 lr 就按照 history 的值来计算
        # 2.输入 lr 为 0，那么 lr 就按照 0 来计算，也就是说背景模型停止更新
        # 3.输入 lr 在 0 ~ 1 之间，那么背景模型更新速度为 lr，lr 越大更新越快，算法内部表现为当前帧参与背景更新的权重越大
        self.lr = lr if lr >= 0 and self.frame_count > 1 else 1 / min(2 * self.frame_count, self.history)

        pool = mp.Pool(int(mp.cpu_count()))
        self.mask = np.zeros(image.shape[:2], dtype=int)
        # 对原图像中的每一行进行并行计算
        result = pool.map_async(self.parallel, [i for i in range(self.image.shape[0])]).get()
        pool.close()
        pool.join()

        # 计算完成之后再进行组合，得到最后的结果
        for row, mask_row, gmm_model_row, mean_model_row, gauss_modes_row in result:
            self.mask[row] = mask_row
            self.gauss_modes[row] = gauss_modes_row
            self.mean_model[row] = mean_model_row
            self.gmm_model[row] = gmm_model_row

        return self.mask

    def parallel(self, row):
        invoker = GuassInvoker(self.image, self.mask, self.gmm_model, self.mean_model, self.gauss_modes, self.nmixtures,
                               self.lr,
                               self.var_threshold, self.background_ratio, self.var_threshold_gen, self.var_init,
                               self.var_min, self.var_max, float(-self.lr * self.ct), self.ct, self.nchannels)
        return invoker.calculate(row)

    def initialize(self, image):
        # gauss_modes 这个矩阵用来存储每一个像素点使用的高斯模型的个数，初始的时候都为 0
        self.gauss_modes = np.zeros(image.shape[:2], dtype=int)
        height, width = image.shape[:2]
        if len(image.shape) == 2:
            self.nchannels = 1
        else:
            self.nchannels = image.shape[2]

        # 高斯混合背景模型分为两部分：
        # 第一部分：height * width * nmixtures (=5) * 3 * sizeof(float)，3 表示包含 weight, variance 以及 c 三个 float 变量，也就是 gmm_model，其中 c 表示和这个高斯模型匹配的个数
        # 第二部分：height * width * nmixtures (=5) * nchannels * sizeof(float)，nchannels 一般为 3，表示 B, G, R 三个变量，其实也就是 mean 每个像素通道均对应一个均值，
        #          刚好有 nchannels 个单位的 float 大小，也就是 mean_model
        # nmixtures = 5 表示高斯模型的数量最多为 5 个
        self.gmm_model = np.zeros((height, width, self.nmixtures, 3), dtype=float)
        self.mean_model = np.zeros((height, width, self.nmixtures, self.nchannels), dtype=float)

    def get_background(self, frame):
        height, width = self.gauss_modes.shape[:2]
        background = np.zeros((height, width, self.nchannels), dtype=np.uint8)

        for row in range(height):
            for col in range(width):
                max_weight = -1
                bg_mean = None

                for mode in range(self.gauss_modes[row, col]):
                    gmm = self.gmm_model[row, col, mode]
                    mean = self.mean_model[row, col, mode]

                    weight = gmm[0]
                    var = gmm[1]

                    # Check if this Gaussian model satisfies the background criteria
                    if weight / (1 - (1 - self.lr) ** gmm[2]) > self.background_ratio:
                        if weight > max_weight:
                            max_weight = weight
                            bg_mean = mean

                if bg_mean is not None:
                    background[row, col] = bg_mean

        return background


if __name__ == '__main__':
    # img = cv2.imread('blank.png')
    cap = cv2.VideoCapture('Sample Input Output/street.mp4')

    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 12.0
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

    # Initialize video writers for foreground and background
    foreground_writer = cv2.VideoWriter(r'Sample Input Output/move1.mp4', fourcc, fps, (width, height))
    background_writer = cv2.VideoWriter(r'Sample Input Output/move2.mp4', fourcc, fps, (width, height))

    mog = GuassMixBackgroundSubtractor()
    frame_count = 0
    while cap.isOpened():
        catch, frame = cap.read()
        frame_count += 1
        if not catch:
            print('The end of the video.')
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = mog.apply(gray).astype('uint8')
            mask = cv2.medianBlur(mask, 3)
            cv2.imwrite('Sample Input Output/mask' + str(frame_count) + '.jpg', mask)
            print('writing mask' + str(frame_count) + '.jpg...')
            # Extract foreground using the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    # Get bounding box for each contour and draw it on the original frame
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract background using the inverse of the mask
            background = mog.get_background(frame)

            # Save original frame, foreground, and background
            cv2.imwrite('Sample Input Output/original' + str(frame_count) + '.jpg', frame)
            cv2.imwrite('Sample Input Output/background' + str(frame_count) + '.jpg', background)

            print(f'Processed frame {frame_count}...')

            foreground_writer.write(mask)
            background_writer.write(background)

    cap.release()
    foreground_writer.release()
    background_writer.release()





