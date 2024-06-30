import cv2 
'''三帧差法实现移动物体检测和前景分离'''
# 视频文件输入初始化 
filename = "Sample Input Output/street.mp4"
camera = cv2.VideoCapture(filename) 

# 获取视频的宽度和高度
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

# 视频文件输出参数设置 
out_fps = 12.0  # 输出文件的帧率 
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2') 
out_binary = cv2.VideoWriter('Sample Input Output/street_gray_foreground.mp4', fourcc, out_fps, (frame_width, frame_height), isColor=False)
out_detect = cv2.VideoWriter('Sample Input Output/street_mvo_detection.mp4', fourcc, out_fps, (frame_width, frame_height))

# 初始化当前帧的前两帧 
lastFrame1 = None
lastFrame2 = None

# 遍历视频的每一帧 
while camera.isOpened(): 
    ret, frame = camera.read() 
    if not ret: 
        break 

    # 如果前两帧是None，对其进行初始化
    if lastFrame1 is None:
        lastFrame1 = frame
        continue
    if lastFrame2 is None:
        lastFrame2 = frame
        continue

    # 计算帧差
    frameDelta1 = cv2.absdiff(lastFrame1, frame) 
    frameDelta2 = cv2.absdiff(lastFrame2, frame) 
    thresh = cv2.bitwise_and(frameDelta1, frameDelta2)
    
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1] 

    # 更新帧队列
    lastFrame1 = lastFrame2
    lastFrame2 = frame.copy()

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_detect = frame.copy()

    for c in cnts:
        if cv2.contourArea(c) < 200: 
            continue 
        (x, y, w, h) = cv2.boundingRect(c) 
        cv2.rectangle(frame_detect, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    out_binary.write(thresh) 
    out_detect.write(frame_detect)

out_binary.release() 
out_detect.release() 
camera.release() 
