'''
这算是个计划之外的小应用，因为开学前还有很多时间，
就把这个在README中说要计划做的东西做出来了
代码实现结果也是识别车牌号，但是代码简易（调用工具包多），
之前做的详细的识别车牌（调用工具包很少，主要靠自己设法处理图像）
这个部分解决问题简单，但是操作中的思考和收获要少，结果也不够精确
./test.png是对原图处理后得到的车牌图片（在程序中把运行结果存到了这里），
程序最终运行结果是print车牌号
前面的几个函数是展示处理后图片的函数，运行中想展示哪一步的图像都可以调用这三个函数展示图像
'''

#运行该程序除了需安装OpenCV外，还需安装imutils、PIL和pytesseract

#导入需要的工具包
import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np
from PIL import Image
import pytesseract

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

#读取图片,彩图如下：
raw_car0 = cv2.imread('car0.png')

# 调整图像大小（防止大分辨率图像操作中出现问题）
car = cv2.resize(raw_car0, (620,480))

#灰度处理
gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

#plt_show1(gray)

#双边滤波（删除多余细节）
gray1 = cv2.bilateralFilter(gray, 13, 15, 15)

#边缘检测
canny = cv2.Canny(gray1, 30, 200)

#寻找轮廓
contours = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)

#选出车牌轮廓
screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018*peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

#遮罩车牌以外的部分
mask = np.zeros(gray1.shape, np.uint8)
new_car = cv2.drawContours(mask, [screenCnt], 0, 255,-1)
new_car = cv2.bitwise_and(car, car, mask = mask)

#截取车牌号图片
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray1[topx : bottomx + 1, topy : bottomy+1]
cv2.imwrite('./test.png',Cropped) #保存图片来展示

#识别字符
text = pytesseract.image_to_string(Cropped, config = '--psm11')
print('车牌号是', text)