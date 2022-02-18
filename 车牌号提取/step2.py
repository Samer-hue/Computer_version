'''
本文件是项目的第二步
需要对第一步得到的图像进一步处理，将每个车牌号的字符图片分离出来，展示并保存到./words这一文件夹
前面的几个函数是展示处理后图片的函数，运行中想展示哪一步的图像都可以调用这三个函数展示图像
一开始主要是载入图片、高斯去噪、灰度处理、自适应阈值处理、轮廓检测和画出轮廓这几步，
对图像初步处理的思路大致和step1中相似，但在具体操作和参数设置上有很多不同，
这些步骤设计和参数选择是我在不断实践中总结的一套比较合适的方法
后面则是筛选出各个字符位置的轮廓（利用长宽比），得到只含一个字符的图片七张，最后把图片保存到./words这一文件夹
'''

#导入需要的工具包
import os
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

#载入图片
raw_img = cv2.imread('./car_license/test.png')

#高斯去噪
gau = cv2.GaussianBlur(raw_img, (3,3), 0)

#灰度处理
gray_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)

#自适应阈值处理,化为二值图像
ret, threshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)

#计算二值图像黑白点的个数，防止遇到绿牌照，让车牌号一直为白色
area_white = 0
area_black = 0
height, width = threshold.shape
for i in range(height):
    for j in range(width):
        if threshold[i,j] == 255:
            area_white += 1
        else:
            area_black += 1

#如果本来白色区域大，进行颜色反转
if area_white > area_black:
    ret, threshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

#把字变胖，让一个字成为一个整体但不同字之间仍然相互独立
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
dilate = cv2.dilate(threshold, kernel)
#plt_show1(dilate)

#轮廓检测
binary, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#画出轮廓
img1 = raw_img.copy()  #复制图片并用此来画出轮廓，防止原图被修改保存
cv2.drawContours(img1, contours, -1, (127, 127, 255), 5)
#plt_show0(img1)

#筛选出各字符位置的轮廓（采用长宽比1.7:1到3.5:1之间，因为‘1’这种数长宽比较大，我把一开始采用的范围扩大了）
words = []
for item in contours:
    word = []
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    word.append(x)
    word.append(y)
    word.append(weight)
    word.append(height)
    words.append(word)

#升序排列
words = sorted(words, key = lambda a:a[0], reverse = False)

#print(words)

#存储每个字符的图片
i = 0
for word in words:
    if(word[3] > (word[2] * 1.7)) and (word[3] < (word[2] * 3.5)):
        i = i+1
        image = raw_img[word[1] : word[1] + word[3], word[0] : word[0] + word[2]]
        plt_show0(image)
        cv2.imwrite('./words/test_'+str(i)+'.png',image)