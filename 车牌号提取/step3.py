'''
第三步需要用到一些模板来对比判断图片中的字符。
模板在./model文件夹中，有上万张样本图片，是我在网络上直接copy的大佬提供的模板。  
通过定义和调用一些对比模板的函数，找出./words中每个图（step2得到并保存的图）最匹配的字符，
并把他们连接起来，就能得到图中的车牌号。  
我把匹配分为两步，第一步匹配汉字，第二步匹配后面的字母和数字，并在其中通过特殊的变量控制这两步的不同执行条件。
这一步因为需要和大量模板匹配，运行起来会比较慢，结果需耐心等待
'''

#导入需要的工具包
import os
from unittest import result
import cv2
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

#索引磨板（车牌号没有I，因为容易和1混淆，同理没有O，因为容易和0混淆）
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
            'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁',
            '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙',]

#读取模板中所有图片的函数
def read(model_name):
    model_list = []
    for filename in os.listdir(model_name):
        model_list.append(model_name + '/' + filename)

    return model_list

#和中文匹配
c_words = []
for i in range(34, 64):
    c_word = read('./model/' + template[i])
    c_words.append(c_word)
k = 1

#定义一个处理图像并匹配车牌号的函数
def read_words(img_path):
    img = cv2.imread(img_path)
    gau = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(gau, cv2.COLOR_RGB2GRAY)
    ret, im = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    best_score = []
    for c_word in c_words:
        score = []
        for word in c_word:
            template_img = cv2.imdecode(np.fromfile(word, dtype = np.uint8), 1)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
            ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
            height, width = template_img.shape
            image = im.copy()
            image = cv2.resize(image, (width, height))
            result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
            score.append(result[0][0])
        best_score.append(max(score))
    if k == 1: #这个判断条件区分匹配汉字还是字母or数字
        return template[34 + (best_score.index(max(best_score)))]
    else:
        return template[(best_score.index(max(best_score)))]

#读取第一个汉字车牌号
a = (read_words('./words/test_1.png')) 
#在k=1时存储汉字字符的值到变量a中，此后将把k改为0，用于调用函数和字母、数字匹配

#和字母、数字匹配
c_words = []
for i in range(0, 34):
    c_word = read('./model/' + template[i])
    c_words.append(c_word)
k = 0

#把结果存储在result中
def show(i):
        return read_words('./words/test_'+str(i)+'.png')

result = a + show(2) +show(3) + show(4) +show(5) + show(6) + show(7)
print('车牌号是: ',result)