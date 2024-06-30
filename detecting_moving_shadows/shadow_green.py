import numpy as np
import cv2 as cv

algo_method = "MOG2"
if algo_method == "MOG2":
    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
else:
    backSub = cv.createBackgroundSubtractorKNN()

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
cap = cv.VideoCapture('Sample Input Output/street.mp4')  # 视频文件 'Sample Input Output/move.mp4'

alpha = 0.4
beta = 0.6  # HSV空间判别阴影的条件之一，[0,1]
t_s = 0.1
t_h = 0.5

while cap.isOpened():  # 循环读取视频帧，直到视频结束或手动退出
    ret, frame = cap.read()  # 读取一帧图像，ret ，bool值，表示是否成功读取一帧图像。
    cv.imshow('Frame_std', frame)
    if ret:
        (SP_B0, SP_G0, SP_R0) = cv.split(frame)  # 将每帧的颜色通道传给二维数组
        fgMask = backSub.apply(frame)
        # 消除掩膜毛刺和细微干扰
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
        deobject = cv.add(frame, frame, mask=fgMask)  # 白色为255,帧差法
        # 获取背景图像
        background = backSub.getBackgroundImage()  # cv.bitwise_and(frame, frame, mask=cv.bitwise_not(fgMask))
        #   在HSV空间判断是否为阴影
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h_f, s_f, v_f = hsv_frame[:, :, 0], hsv_frame[:, :, 1], hsv_frame[:, :, 2]
        # 归一化饱和度值
        s_f = s_f / 255.0
        h_f = h_f / 360.0
        hsv_background = cv.cvtColor(background, cv.COLOR_BGR2HSV)
        h_b, s_b, v_b = hsv_background[:, :, 0], hsv_background[:, :, 1], hsv_background[:, :, 2]
        s_b = s_b / 255.0
        h_b = h_b / 360.0

        # 计算两个矩阵对应位置的绝对差值和相反数的绝对差值
        diff_hf_hb = np.abs(h_f - h_b)
        inverse_diff_hf_hb = 1 - np.abs(h_f - h_b)

        # 创建一个新矩阵，取两个操作结果对应位置的较小值
        new_matrix = np.minimum(diff_hf_hb, inverse_diff_hf_hb)
        # HSV空间影子判断条件
        mask1 = new_matrix <= t_h
        mask2 = s_f - s_b <= t_s
        mask3 = alpha <= v_f / v_b
        mask4 = v_f / v_b <= beta

        mask = mask1 & mask2 & mask3 & mask4

        # 将影子部分变为绿色
        SP_B0[mask] = 0
        SP_G0[mask] = 255
        SP_R0[mask] = 0


        frame[:, :, 0] = SP_B0[:, :]  # B通道
        frame[:, :, 1] = SP_G0[:, :]  # G通道
        frame[:, :, 2] = SP_R0[:, :]  # R通道
        cv.imshow('shadow_green', frame)

        keyboard = cv.waitKey(20)  # 等待用户按键输入，设置延迟时间为20毫秒
        if keyboard & 0xff == ord('q'):  # 显然只需要低8位包含asc码的有效信息
            break  # 退出循环
    else:
        break

cap.release()  # 释放视频对象
cv.destroyAllWindows()  # 关闭所有窗口