import csv
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from statsmodels.tsa._stl import STL
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("project_detect").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")   # 设置日志级别
spark = SparkSession(sc)
ipath = "/home/hadoop/Experiment/project/"

def find(ans, l):
    dot = 0
    T = 0
    ans1 = []
    for i in range(len(ans)):
        ans1.append(int(ans[i]/l))
    print(ans1)
    ans2 = []
    flag = -1
    repet = []
    unrepet = []
    for i in range(len(ans1)):
        if ans1[i] in unrepet:
            if ans1[i] in repet:
                continue
            else:
                repet.append(ans1[i])
        else:
            unrepet.append(ans1[i])
    T = len(repet)
    dot = len(unrepet)-len(repet)
    return dot, T


def STL1(addre, name):
    elecequip = read_csv(addre, usecols=[0])  # ,index_col='time'
    stl = STL(elecequip.values, period=24)
    # print(stl.values)
    # print(stl.fit().trend)#趋势/增幅
    stl.fit().plot()
    # print(stl.fit().seasonal)#季节/周期
    residual = list(stl.fit().resid)  # 残差
    print(residual)
    Q1 = np.percentile(residual, 25)
    Q3 = np.percentile(residual, 75)
    IQR = Q3-Q1
    up_bound = []
    low_bound = []
    x = []
    ans = []  # 记录异常点
    residuale = [0 for j in range(len(residual))]
    print(elecequip.values.shape[0])
    print(len(residual))
    for i in range(len(residual)):
        x.append(i)
        up_bound.append(Q3+1.5*IQR)
        low_bound.append(Q1-1.5*IQR)
        # print(residual[0,i])
        # print(residual)
        if (Q3+1.5*IQR) <= residual[i]:
            residuale[i] = residual[i]
            ans.append(i)
        elif (Q1-1.5*IQR) >= residual[i]:
            residuale[i] = residual[i]
            ans.append(i)
    print(elecequip.values.shape[0])
    print(ans)
    print(len(residuale))
    # plt.show()
    # plt.savefig('results1\\' + '1' +'.png')
    plt.plot(x, up_bound,'m--')
    plt.plot(x, low_bound,'m--')
    # plt.plot(x, residuale,'o')
    # plt.show()  #显示异常
    plt.savefig(ipath+ 'detect_result/' + str(name) +'.png')
    dot, T = find(ans, 24)
    print(dot)
    print(T)
    return dot, T


def abnormal(addr):
   # ['小区内的平均用户数'], ['小区 PDCP 流'], ['平均激活用户数']
    rows = []
    addr1 = ipath + '小区平均用户/小区'+addr
    dot1, T1 = STL1(addr1, str(addr)+'-avUser')
    row1 = ['小区内的平均用户数', 24, dot1, T1]
    rows.append(row1)
    addr2 = ipath + '小区PDCP流量/小区PDCP'+addr
    dot1, T1 = STL1(addr2, str(addr)+'-PDCP')
    row1 = ['小区PDCP流量', 24, dot1, T1]
    rows.append(row1)
    addr3 = ipath + '小区平均激活用户数/小区激活'+addr
    dot1, T1 = STL1(addr3, str(addr)+'-active')
    row1 = ['平均激活用户数', 24, dot1, T1]
    rows.append(row1)
    

def main():
    for i in range(1,59):
        a = 26019000+i
        add = str("%s.csv" % (a))
        abnormal(add)

main()
