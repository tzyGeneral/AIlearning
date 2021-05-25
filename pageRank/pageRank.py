# -*- coding: utf-8 -*-
from pygraph.classes.digraph import digraph


class PRIterator:
    __doc__ = """计算一张图中的PR值"""

    def __init__(self, dg):
        self.damping_factor = 0.85  # 阻尼系数
        self.max_iterations = 100  # 最大迭代次数
        self.min_delta = 0.00001  # 判断迭代是否结束的参数
        self.graph = dg

    def page_rank(self):
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    digraph.add_edge(self.graph, (node, node2))

        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}
        # 给每个节点赋予初始的PR值，第一轮的PR值是均等的，即 1/N
        page_rank = dict.fromkeys(nodes, 1.0 / graph_size)
        damping_value = (1.0 - self.damping_factor) / graph_size

        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                # 遍历所有"入射"的页面
                for incident_page in self.graph.incidents(node):
                    # "入射"页面的权重更具其出链个数均分，然后传递给当前页面
                    rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
                # 增加随机概率转移矩阵部分
                rank += damping_value
                change += abs(page_rank[node] - rank)  # 绝对值
                page_rank[node] = rank
            print("This is NO.%s iteration" % (i + 1))
            print(page_rank)

            if change < self.min_delta:
                flag = True
                break
        if flag:
            print("Finished in %s iterations" % node)
        else:
            print("Finished out of 100 iterations")
        return page_rank


if __name__ == '__main__':
    # 创建一个网络拓朴图
    dg = digraph()

    # 这里添加所有"用户"的节点
    dg.add_nodes(["A", "B", "C", "D", "E"])
    # 这里添加 节点关系
    dg.add_edge(("A", "B"))
    dg.add_edge(("A", "C"))
    dg.add_edge(("A", "D"))
    dg.add_edge(("B", "D"))
    dg.add_edge(("C", "E"))
    dg.add_edge(("D", "E"))
    dg.add_edge(("B", "E"))
    dg.add_edge(("E", "A"))

    # PRrank迭代计算
    pr = PRIterator(dg)
    page_ranks = pr.page_rank()
    print(page_ranks)