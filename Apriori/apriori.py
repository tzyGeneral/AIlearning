# coding: utf8
from numpy import *


def loadDataSet():
    """
    加载数据集
    :return:
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    创建集合C1
    :param dataSet: 原始数据集
    :return: frozenset: 返回一个 frozenset 格式的 list
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 遍历所有的元素，如果不在C1中出现过，那么就 append
                C1.append([item])
    # 对数组进行 从小到大 排序
    C1.sort()
    # frozenset 表示冻结的 set 集合，元素无法被改变，可以把它当作字典的key来使用
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    """
    scanD(计算候选数据 Ck 在 数据集 D 中的支持度，并返回支持度大于最小支持度 minSupport 的数据)
    :param D: 数据集
    :param Ck: 候选项集列表
    :param minSupport: 最小支持度
    :return:
        reList：支持度大于minSupport的集合
        supportData：候选项集支持度数据
    """
    # ssCnt 临时存放候选数据集 Ck 的频率，例如: a->10, b->5, c->8
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # s.issubset(t) 测试是否 s 中每一个元素都在t中
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # 数据集D的数量
    retList = []
    supportData = []
    for key in ssCnt:
        # 支持度 = 候选项(key)出现的次数 / 所有数据集的数量
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 在 retList 的首位插入元素，只支持存储满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    return retList, supportData


def aprioriGen(LK, k):
    """
    aprioriGen 输入频繁集列表 LK 和返回的元素个数 k， 然后输出候选项集 Ck
    例如：以{0},{1},{2} 为输入且 k=2 则输出 {0,1},{0,2},{1,2}. 以 {0,1},{0,2},{1,2} 为输入且k=3 则输出 {0,1,2}
    仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作
    这是一个更高效的算法
    :param LK: 频繁项集列表
    :param k: 返回的项集元素个数（若元素的前k-2相同，就进行合并）
    :return:
        retList: 元素两两合并的数据集
    """
    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(LK[i])[: k-2]
            L2 = list(LK[j])[: k-2]

            L1.sort()
            L2.sort()

            # 第一次 L1，L2为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    """
    首先构建集合C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求，
    那么满足最小支持度的项集构成集合构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，
    然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。）
    :param dataSet: 原始数据集
    :param minSupport:支持度的阈值
    :return:
        L：频繁项集的全集
        supportData：所有元素和支持度的全集
    """
    # C1 即对 dataSet 进行去重，排序，放入list中，然后转换所有的元素为frozenset
    C1 = createC1(dataSet)
    # 对每一行进行set转换，然后存放到集合中
    D = map(set, dataSet)
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)

    # L 加了一层 List，L一共2层List
    L = [L1]
    k = 2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。
    # L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while (len(L[k-2]) > 0):
        # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        Ck = aprioriGen(L[k-2], k)

        Lk, supK = scanD(D, Ck, minSupport)  # 计算候选数据集Ck在数据集D中的支持度，并返回支持度大于minSupport的数据
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break

        # Lk 表示满足频繁子项的集合，L元素在增加，例如
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1

    return L, supportData


def calcConf(freqSet, H, supportData, brl, minConf=0.7)
    """
    计算可信度
    对两个元素的频繁项，计算可信度，例如：{1,2}/{1} 或者 {1,2}/{2} 看是否满足条件
    :param freqSet: 频繁项集中的元素，例如：frozenset([1,3])
    :param H: 频繁项集中的元素的集合，例如：[frozenset([1]), frozenset([3])]
    :param supportData: 所有元素的支持度的字典
    :param brl: 关联规则列表的空数组
    :param minConf: 最小的可信度
    :return: 
        prunedH：记录 剋下年度大于阈值的集合
    """
    # 记录剋下年度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H:  # 假设 freqSet = frozenset([1, 3]), H = [frozenset([1]), frozenset([3])]，那么现在需要求出 frozenset([1]) -> frozenset([3]) 的可信度和 frozenset([3]) -> frozenset([1]) 的可信度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)

    return prunedH


# 递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    :param freqSet: 频繁项集中的元素，例如: frozenset([2, 3, 5])
    :param H: 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
    :param supportData: 所有元素的支持度的字典
    :param brl: 关联规则列表的数组
    :param minConf: 最小可信度
    :return:
    """
    # H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，长度由 aprioriGen(H, m+1) 这里的 m + 1 来控制
    # 该函数递归时，H[0] 的长度从 1 开始增长 1 2 3 ...
    # 假设 freqSet = frozenset([2, 3, 5]), H = [frozenset([2]), frozenset([3]), frozenset([5])]
    # 那么 m = len(H[0]) 的递归的值依次为 1 2
    # 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5]) ，没必要再计算 freqSet 与 H[0] 的关联规则了。
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # 生成 m+1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]), frozenset([3]), frozenset([5])]
        # 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
        # 第二次 。。。没有第二次，递归条件判断时已经退出了
        Hmp1 = aprioriGen(H, m+1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print ('Hmp1=', Hmp1)
        print ('len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet))
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    """
    :param L: 频繁项集列表
    :param supportData: 频繁项集支持度的字典
    :param minConf: 最小置信度
    :return:
        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    """
    bigRuleList = []
    # 假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            # 假设：freqSet= frozenset([1, 3]), H1=[frozenset([1]), frozenset([3])]
            # 组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            # 2 个的组合，走 else, 2 个以上的组合，走 if
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def getActionIds():
    from time import sleep
    from votesmart import votesmart

    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('data/11.Apriori/recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) # api call
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print ('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print ("problem getting bill %d" % billNum)
        sleep(1)                                      # delay to be polite
    return actionIdList, billTitleList



def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print ("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning


def testApriori():
    # 加载测试数据集
    dataSet = loadDataSet()
    print('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.7)
    print('L(0.7): ', L1)
    print('supportData(0.7): ', supportData1)

    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    # Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori(dataSet, minSupport=0.5)
    print('L(0.5): ', L2)
    print('supportData(0.5): ', supportData2)


def testGenerateRules():
    # 加载测试数据集
    dataSet = loadDataSet()
    print('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.5)
    print('L(0.7): ', L1)
    print('supportData(0.7): ', supportData1)

    # 生成关联规则
    rules = generateRules(L1, supportData1, minConf=0.5)
    print('rules: ', rules)


def main():
    # 测试 Apriori 算法
    # testApriori()

    # 生成关联规则
    # testGenerateRules()

    ##项目案例
    # # 构建美国国会投票记录的事务数据集
    # actionIdList, billTitleList = getActionIds()
    # # 测试前2个
    # transDict, itemMeaning = getTransList(actionIdList[: 2], billTitleList[: 2])
    # transDict 表示 action_id的集合，transDict[key]这个就是action_id对应的选项，例如 [1, 2, 3]
    # transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    # # 得到全集的数据
    # dataSet = [transDict[key] for key in transDict.keys()]
    # L, supportData = apriori(dataSet, minSupport=0.3)
    # rules = generateRules(L, supportData, minConf=0.95)
    # print (rules)

    # # 项目案例
    # # 发现毒蘑菇的相似特性
    # # 得到全集的数据
    dataSet = [line.split() for line in open("data/11.Apriori/mushroom.dat").readlines()]
    L, supportData = apriori(dataSet, minSupport=0.3)
    # # 2表示毒蘑菇，1表示可食用的蘑菇
    # # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print(item)

    for item in L[2]:
        if item.intersection('2'):
            print(item)


if __name__ == "__main__":
    main()

