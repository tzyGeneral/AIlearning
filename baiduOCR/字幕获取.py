# -*- coding = UTF-8 -*-
import base64
import cv2
import requests
from aip import AipOcr


imgPath = 'D:/pycharmProject2/AIlearning/baiduOCR/notDone/'
newImgPath = 'D:/pycharmProject2/AIlearning/baiduOCR/done/'

def requestApi(img):
    """
    访问百度API获取字幕内容
    :param img:
    :return:
    """
    requestUrl = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    params = {"image": img, "language_type": 'CHN_ENG'}
    access_token = "24.8c1eda6ad205c2bef34a1f98263f20a3.2592000.1592970551.282335-20049672"
    request_url = requestUrl + "?access_token=" + access_token
    headers = {"content-type": "application/x-www-form-urlencoded"}
    response = requests.post(request_url, data=params, headers=headers)
    result = response.json()
    return result


def get_file_content(filePath: str):
    """
    读取一张图片并转换为b64encode编码格式
    :param filePath:
    :return:
    """
    with open(filePath, 'rb') as f:
        return base64.b64encode(f.read())


def isChinese(word):
    """
    判断是否为中文
    :param word:
    :return:
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def tailor(fileName: str):
    """
    切割字幕区域，进行二值化
    :param imgPath:
    :return:
    """
    filePath = imgPath + fileName
    img = cv2.imread(filePath)
    hegiht = len(img)
    width = len(img[0])

    # cropped = img[int(hegiht - hegiht/5):hegiht, 0:width]  # 裁剪坐标为[y0:y1, x0:x1]
    cropped = img[0:int(hegiht / 5), 0:width]  # 裁剪坐标为[y0:y1, x0:x1]
    imgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    thresh = 200
    ret, binary = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)  # 输入灰度图，输出二值图
    binary1 = cv2.bitwise_not(binary)  # 取反
    cv2.imwrite(newImgPath + fileName, binary1)


def subtitle(fileName: str):
    # 定义一个列表存放words
    array = []
    tailor(fileName)
    image = get_file_content(newImgPath + fileName)
    try:
        result = requestApi(image)['words_result']
        for item in result:
            if isChinese(item['words']):
                print(item['words'])
                array.append(item['words'])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    fileName = 'test.png'
    subtitle(fileName)

