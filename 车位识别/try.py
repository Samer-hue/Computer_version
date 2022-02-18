'''
这个程序是为了识别停车场中心的车位，并判断车位是被占据还是空闲，统计车位空闲数和总数
是我在一些参考一些源码后根据自己需求修改的程序，识别的视频用了教程中的实例视频
结果展示是识别后的视频，视频播放完成自动退出，也可按Q键退出
这算是半成品，功能基本实现，只能识别图中一部分车位，目前仍在优化问题
我截取视了频中的几个图像作为结果展示放在文件夹./result_img中
前面的几个函数是展示处理后图片的函数，运行中想展示哪一步的图像都可以调用这三个函数展示图像
'''

#导入所需工具包
import cv2
import pickle
import cvzone
import numpy as np
from matplotlib import pyplot as plt

#显示图片
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#展示彩色图片
def plt_show0(img):
    plt.imshow(img[:,:,[2,1,0]])
    plt.show()
 

#展示灰度图片
def plt_show1(img):
    plt.imshow(img, cmap = 'gray')
    plt.show()

#导入视频
cap = cv2.VideoCapture('parking_video.mp4')

width, height = 25, 10

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

#遍历，处理视频
def checkParkingSpace(imgPro):

    spaceCounter = 0

    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y+height, x:x+width]
        count = cv2.countNonZero(imgCrop)
        cvzone.putTextRect(img, str(count), (x,y+height), scale=0.3, thickness=1, offset=0)

        if count < 100:
            color = (0, 255, 0)
            thickness = 2
            spaceCounter+=1
        else:
            color = (0,0,255)
            thickness = 1

        cv2.rectangle(img,pos,(pos[0] + width,pos[1] + height),color,thickness)

        cvzone.putTextRect(img, f'Free:{spaceCounter}/{len(posList)}', (100,50), scale=5, 
        thickness=5, offset=20, colorR=(0,200,0))


while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    if cap.isOpened():
        success, img = cap.read()
    else:
        success = False
        
    while success:
        success, img = cap.read()
        if img is None:
            break
        if success == True:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations = 1)

            checkParkingSpace(imgDilate)

            cv2.imshow('show(press Q to quit)',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()