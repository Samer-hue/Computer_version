'''
如果想运行该程序，
先创建一个包含prototxt文件、caffemodel文件和python文件（即我文件夹里除了图片之外的所有文件）的文件夹
在cmd中跳转到这个文件夹所在的位置，再运行以下指令，即可通过你的电脑摄像头来识别人脸
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
运行时，按'q'键可退出检测界面
我的执行结果截图可以在这个文件夹下的运行结果.jpg中看到
'''

'''
程序是在参考了一些大佬的代码后根据需求修改的
程序的主要步骤是解析参数、初始化列表及颜色集、加载训练好的模型、初始化视频流和FPS计数器、对每一帧进行遍历和展示、
检查quit键、更新 fps 计数器、停止定时器，并显示FPS信息、清理
'''

#导入用到的工具包
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#初始化类列表和颜色集（初始化 CLASS 标签和相应的随机 COLORS）
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#加载训练好的模型，并设置视频流
print("加载模型中……请稍等")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#初始化视频流和FPS计数器
print("打开视频系统中……请稍等")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
print('按Q键退出')

#遍历每一帧
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()

	#看置信度的值，判断我们能否在目标周围绘制边界框和标签
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = '{}: {:.2f}%'.format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	#展示帧
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#检查 quit 键
	if key == ord("q"): #运行时，按'q'键可退出检测界面
		break

	#更新 fps 计数器
	fps.update()

#停止定时器，并显示FPS信息
fps.stop()
print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

#清理
cv2.destroyAllWindows()
vs.stop()