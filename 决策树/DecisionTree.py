





def createDataSet():
    """
    创建数据集
    :return:
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算给定数据的香农熵
    :param dataSet:
    :return:
    """
    # 求list的长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    # 下面输出我们测试的数据集的一些信息

    # 计算分类标签label出现的次数



def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()

    # 计算label分类标签的香农熵
    calcShannonEnt(myDat)