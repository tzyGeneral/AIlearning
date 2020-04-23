from numpy import *


# 默认数据是用tab分隔，并且是数值类型
def loadDataSet(fileName):
    """
    解析每一行，并转换未float数据（该函数读取一个以tab键为分隔符的文件，然后将每行的内容保存为一组浮点数）
    :param fileName: 文件名
    :return:
        dataMat: 每一行的数据集的array类型
    """
    # 假定最后一列是结果值
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每一行转换为浮点数
        fltLine = [float(x) for x in curLine]
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    将数据集，按照feature列的value进行二元切分（在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集切分得到两个子集并返回）
    :param dataSet: 数据集
    :param feature: 待切分的特征列
    :param value: 特征要比较的值
    :return:
        mat0: 小于value的数据集再左边
        mat1: 大于value的数据集再右边
    """
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1


# 返回每一个叶子节点的均值
# 我的理解是：legLeaf是尝试叶节点的函数，就是求均值，即用聚类中心点来代表这类数据
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 计算总方差=方差*样本数
# 我的理解是：求这组数据的安方差，即通过决策树划分，可以让靠近的数据分到同一类去
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 用最佳方式切分数据集
# 生成相应的叶节点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    用最佳方式切分数据集 和 生成相应的叶节点
    :param dataSet: 加载的原始数据集
    :param leafType: 建立叶子节点的函数
    :param errType: 误差计算函数（求总方差）
    :param ops: [容许误差下降值，切分的最少样本树]
    :return:
        bestIndex feature的index坐标
        bestValue: 切分的最优值
    """
    # ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
    # 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
    # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
    tolS = ops[0]
    # 划分最小size小于，就不继续划分了
    tolN = ops[1]
    # 如果数据集的最后一列所有值相等就退出
    # dataSet[:, -1].T.tolist()[0] 取数据集的最后一列，转置为行向量，然后转换为list,取该list中的第一个元素。
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 如果集合size为1，也就是说全部的数据都是同一个类别，不用继续划分
        return None, leafType(dataSet)

    # 计算行列值
    m, n = shape(dataSet)
    # 无分类误差的总方差和
    S = errType(dataSet)
    # inf正无穷大
    bestS, bestIndex, bestValue = inf, 0, 0
    # 循环处理每一列对应的feature值
    for featIndex in range(n-1):  # 对于每个特征
        # 下面的一行表示的是将某一列全部的数据转换为行，然后设置为list形式
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 对该列进行分组，然后组内的成员的val值进行二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小值
            # 如果划分后误差小于 bestS, 则说明找到了新的bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 判断二元切分的方式的元素误差是否符合预期
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 对整体的成员进行判断，是否符合预期
    # 如果集合的 size 小于 tolN
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 当最佳划分后，集合过小，也不划分产生叶节点
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 假设dataSet是numpy mat类型的，那么我们可以进行array过滤
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    获取回归树（递归函数：如果构建的是回归树，该模型是一个常数，如果是模型树，其模型是一个线性方程）
    :param dataSet: 加载的原始数据集
    :param leafType: 建立叶子点的函数
    :param errType: 误差计算函数
    :param ops: [容许误差下降值，切分的最少数据样本]
    :return:
        retTree  决策树最后的结果
    """
    # 最好的切分方式：feature索引值，最优切分值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果splitting 达到一个停止条件，那么返回val
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 大于在右边，小于在左边，分为2个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归的进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 判断节点是否是一个字典
def isTree(obj):
    """
    测试输入变量是否是一棵树，即是否是一个字典
    :param obj: 输入变量
    :return:
    """
    return (type(obj).__name__ == 'dict')


# 计算左右枝丫的均值
def getMean(tree):
    """
    从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的均值
    对 tree 进行坍塌处理，即返回树平均值
    :param tree: 输入的树
    :return: 返回 tree 节点的平均值
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']) / 2.0


# 检查是否适合合并分支
def prune(tree, testData):
    """
    从上而下找到叶节点，用测试数据集从判断这些叶节点合并能否降低测试误差
    :param tree: 待剪枝的树
    :param testData: 剪枝所需要的测试数据 testData
    :return:
        tree 剪枝完成的树
    """
    # 判断是否册数数据集没有数据，如果没有，就直接返回tree本身的均值
    if shape(testData)[0] == 0:
        return getMean(tree)

    # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    # 如果是左边分支是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)

    # 如果是右边分支是字典，就传入右边的数据集和右边的分枝，进行递归
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    # 1. 如果正确
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # power(x, y)表示x的y次方
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 如果 合并 的总方差 < 不合并的总方差，那么进行合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 得到模型的ws系数：f(x) = x0 + x1*feature1 + x3*feature2 ...
def modelLeaf(dataSet):
    """
    当数据不需要再切分的时候，生成叶节点的模型。
    :param dataSet: 输入数据集
    :return:
        调用 linearSolve函数，返回得到的回顾系数ws
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


# 计算线性模型的误差值
def modelErr(dataSet):
    """
    再给定数据集上计算误差
    :param dataSet: 输入数据集
    :return:
        调用linearSolve函数，返回yHat 和 Y之间的平方五误差。
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def linearSolve(dataSet):
    """
    将数据集格式化成目标变量Y和自变量X，执行简单的线性回顾，得到ws
    :param dataSet: 输入数据集
    :return:
        ws：执行线性回归的回归系数
        X：格式化自变量X
        Y：格式化目标变量Y
    """
    m, n = shape(dataSet)
    # 产生一个关于1的矩阵
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    # X的0列为1，常数项，用于计算平衡误差
    X[:, 1: n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]

    # 转置为矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的逆不存在，会造成程序异常
    if linalg.det(xTx) == 0.0:
        raise NameError('这个矩阵时奇异的, 不能为逆,\ntry increasing the second value of ops')
    # 最小二乘法求最优解：w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 回归树测试案例
# 为了和 modelTreeEval（）保持一致，保留两个输入参数
def regTreeEval(model, inDat):
    """
    对回归树进行预测
    :param model: 指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
    :param inDat:输入的测试数据
    :return:
        float(model) -- 将输入的模型数据转换为 浮点数 返回
    """
    return float(model)


# 回归树测试案例
# 对输入数据进行格式化处理，在原数据机制上增加第0列，元素的值都是1
# 也就是增加偏移值，和我们之前简单的线性回归时一个套路，增加一个偏移量
def modelTreeEval(model, inDat):
    """
    对 模型树 进行预测
    :param model: 输入模型，可选值为回归树模型或者模型树模型，这里为模型树模型
    :param inDat: 输入的测试数据
    :return:
        float(X * model): 将测试数据乘以回顾系数得到一个预测值，转化为浮点数返回
    """
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    return float(X * model)


# 计算预测的结果
# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
# modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
# 调用modelEval()函数，该函数的默认值为regTreeEval()
def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    对特定模型的树进行预测，可以时回归树 也可以是 模型树
    :param tree: 已经训练好的树的模型
    :param inDat: 输入测试数据
    :param modelEval: 预测的树的模型类型，可选值为regTreeEval(回归树) 或 modelTreeEval(模型树)，默认为回归树
    :return:
        返回预测值
    """
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] <= tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 预测结果
def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    调用treeForeCast，对特定模型的树进行预测，可以是 回归树 也可以是 模型树
    :param tree:  已经训练好的树的模型
    :param testData:  输入的测试数据
    :param modelEval:  预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    :return:
        返回预测值矩阵
    """
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == "__main__":
    # 回归树 VS 模型树 VS 线性回归
    trainMat = mat(loadDataSet('./data/bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('./data/bikeSpeedVsIq_test.txt'))

    # 回归树
    myTree1 = createTree(trainMat, ops=(1, 20))
    print(myTree1)
    yHat1 = createForeCast(myTree1, testMat[:, 0])
    print("回归树:", corrcoef(yHat1, testMat[:, 1], rowvar=False)[0, 1])

    # 模型树
    myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    print(myTree2)
    print("模型树:", corrcoef(yHat2, testMat[:, 1], rowvar=False)[0, 1])

    # 线性回归
    ws, X, Y = linearSolve(trainMat)
    print(ws)
    m = len(testMat[:, 0])
    yHat3 = mat(zeros((m, 1)))
    for i in range(shape(testMat)[0]):
        yHat3[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    print("线性回归:", corrcoef(yHat3, testMat[:, 1],rowvar=False)[0, 1])
