'''
本文件完成项目第一步，把所给图片中的车牌区域提取出来，展示，并保存到./car_license/test.png
前面的几个函数是展示处理后图片的函数，运行中想展示哪一步的图像都可以调用这三个函数展示图像
主要是载入图片、灰度处理、高斯滤波、sobel边缘检测、自适应阈值处理、闭运算、x轴腐蚀再膨胀，y轴膨胀再腐蚀、
中值滤波、最后检测并绘制轮廓这些步骤，通过车牌轮廓外接长方形特有的长宽比特征筛选出车牌所在区域，
最后保存和展示得到的车牌区域图片，保存好的图片将在第二步（step2.py）中用到
'''

#导入需要的工具包
import cv2
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

#读取图片并转化为灰度图,彩图如下：
raw_car0 = cv2.imread('car0.png')

#灰度图如下：
car0 = cv2.cvtColor(raw_car0, cv2.COLOR_BGR2GRAY)


#高斯滤波去噪
gau = cv2.GaussianBlur(car0, (3,3), 0)


#利用sobel算子进行边缘检测
sobel_x = cv2.Sobel(gau, cv2.CV_16S, 1, 0)
absx = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.Sobel(gau, cv2.CV_16S, 0, 1)
absy = cv2.convertScaleAbs(sobel_y)
dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

#自适应阈值处理（把边缘检测后的图片转化为只有黑、白两种颜色的图片）
ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)

#闭运算（把白色部分连成一个整体）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,5))
car = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations = 1)


#去除小白点
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (1,19))

#X轴：膨胀再腐蚀
car = cv2.dilate(car,kernelx)
car = cv2.erode(car,kernelx)

#Y轴：腐蚀再膨胀
car = cv2.dilate(car,kernely)
car = cv2.erode(car,kernely)

#中值滤波(进一步去除噪音点)
car = cv2.medianBlur(car, 15)

#检测和画出轮廓
binary, contours, hierarchy = cv2.findContours(car, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
acar = raw_car0.copy() #复制图片并用此来画出轮廓，防止原图被修改保存
dcar = cv2.drawContours(acar, contours, -1, (255, 0, 0), 5)

#筛选出车牌位置（利用长宽比）
for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if (weight > (3*height)) and (weight < (4*height)):
        car = raw_car0[y:y+height, x:x+weight]
        #保存图像
        cv2.imwrite('./car_license/test.png',car)


#展示图片
show = cv2.imread('./car_license/test.png')
plt_show0(show)
