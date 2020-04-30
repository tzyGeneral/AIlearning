import re

def loadDataSet(fileName):
    """
    读取文件
    :param fileName:
    :return:
    """
    jiamiCount = 0
    noJiaMiCount = 0
    find500 = 0
    fr = open(fileName)
    for lineArr in fr.readlines():
        pattern = re.compile('HTTP/1.1" (\d+)')
        status = pattern.findall(lineArr)
        if status:
            if status[0] != '200':
                print(lineArr)
                print(status)

        if 'ko=219' in lineArr:
            if 'getWea5' in lineArr:
                jiamiCount += 1
            if 'getWeaXI' in lineArr:
                noJiaMiCount += 1

    print('加密接口调用',str(jiamiCount))
    print('未加密接口调用',str(noJiaMiCount))
    print('出现500错误', str(find500))


# if __name__ == "__main__":
#     loadDataSet('./weatherSSL_nginx_access.log')
import hashlib

md5 = hashlib.md5('4327b86tg72b0ac1g0ko362l0_tt=158790'.encode('utf-8'))
print(md5.hexdigest().upper() == '6CF66622F54BB52A26BD6B13705A4D12')