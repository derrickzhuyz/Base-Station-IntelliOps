import csv
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from statsmodels.tsa._stl import STL
from statsmodels.tsa._stl import STL
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("project_predict").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
spark = SparkSession(sc)
ipath = "/home/hadoop/Experiment/project/"


def normal(data):
    data = data.T
    data_noraml=[]
    for i in range(data.shape[0]):
        a=data[i]
        a=np.mat(a.values)
        max_a=a.max()
        min_a =a.min()
        a_normal=[]
        for j in range(a.shape[1]):
            a_normal.append(float((a[:,j]-min_a)/(max_a-min_a)))
        data_noraml.append(a_normal)
    data_noraml=np.mat(data_noraml)
    return data_noraml.T

def loaddataset(filename):
    fp = open(filename)
    dataset = []
    labelset = []
    for i in fp.readlines():
        a = i.strip().split()
        # 存储属性数据
        dataset.append([float(j) for j in a[:len(a) - 1]])
        # 存储标签数据
        labelset.append(int(float(a[-1])))
    return dataset, labelset


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def trainning(dataset, labelset, test_data, test_label):
    # 将列表转化为矩阵
    data = np.mat(dataset)
    label = np.mat(labelset)
    # 初始化参数w
    w = np.ones((len(dataset[0]) + 1, 1))
    a = np.ones((len(dataset), 1))
    #print(a)
    #print(data)
    data = np.c_[data, a]
    # 步长
    n = 0.0001
    # 每次迭代计算一次正确率（在测试集上的正确率）
    # 达到0.75的正确率，停止迭代
    rightrate = 0.0
    while rightrate < 0.75:
        # 计算当前参数w下的预测值
        c = sigmoid(np.dot(data, w))
        b = c - label
        change = np.dot(np.transpose(data), b)
        w = w - change * n
        # 预测，更新正确率
        rightrate = test(test_data, test_label, w)
    F = F1(test_data, test_label, w)
    return w,F


def test(dataset, labelset, w):
    data = np.mat(dataset)
    a = np.ones((len(dataset), 1))
    data = np.c_[data, a]
    # 使用训练好的参数w进行计算
    y = sigmoid(np.dot(data, w))
    b, c = np.shape(y)
    # 记录预测正确的个数，用于计算正确率
    rightcount = 0
    for i in range(b):
        # 预测标签
        flag = -1
        # 正例
        if y[i, 0] > 0.5:
            flag = 1
        # 反例
        else:
            flag = 0
        # 记录预测正确的个数
        if labelset[i] == flag:
            rightcount += 1
    # 正确率
    rightrate = rightcount / len(dataset)
    return rightrate


def F1(dataset, labelset, w):
    data = np.mat(dataset)
    a = np.ones((len(dataset), 1))
    data = np.c_[data, a]
    # 使用训练好的参数w进行计算
    y = sigmoid(np.dot(data, w))
    b, c = np.shape(y)
    # 记录预测正确的个数，用于计算正确率
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(b):
        # 预测标签
        flag = -1
        # 大于0.5的为正例
        if y[i, 0] > 0.5:
            flag = 1
        # 小于等于0.5的为反例
        else:
            flag = 0
        if labelset[i] == flag and labelset[i]==0:
        # 记录预测正确的个数
            TP = TP+1
        elif labelset[i] == flag and labelset[i]==1:
            TN = TN + 1
        elif labelset[i] != flag and labelset[i]==1:
            FP = FP + 1
        else:
            FN = FN + 1
    # 正确率
    F=2*TP/(b+TP-TN+1)
    return F


def predic(dataset,  w):
    data = np.mat(dataset)
    a = np.ones((len(dataset), 1))
    data = np.c_[data, a]
    # 使用训练好的参数w进行计算
    y = sigmoid(np.dot(data, w))
    b, c = np.shape(y)
    ans=[]
    for i in range(b):
        # 预测标签
        flag = -1
        # 大于0.5的为正例
        if y[i, 0] > 0.5:
            flag = 1
        # 小于等于0.5的为反例
        else:
            flag = 0
        ans.append(flag)
        # 记录预测正确的个数
    # 正确率
    #rightrate = rightcount / len(dataset)
    return ans

def STL1(addre,X,y):
    #print(len(X[0]))
    elecequip = read_csv(addre,usecols=[0])#,index_col='time'
    stl = STL(elecequip.values, period=24)
    residual=list(stl.fit().resid) #残差
    Q1=np.percentile(residual,25)
    Q3=np.percentile(residual,75)
    IQR=Q3-Q1
    up_bound=[]
    low_bound=[]
    x=[]
    X = X.tolist()
    y1=[]
    for i in range(len(residual)):
        x.append(i)
        up_bound.append(Q3+1.5*IQR)
        low_bound.append(Q1-1.5*IQR)
        if (Q3+1.5*IQR)<=residual[i]:
            y1.append(1)
        elif (Q1-1.5*IQR)>=residual[i]:
            y1.append(1)
        y1.append(0)
    size=len(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X[:size-24], y1[:size-24], test_size=0.25, random_state=33)
    ansx= X[-24:]
    w ,F= trainning(X_train, Y_train,X_test, Y_test)
    rightrate = predic(list(ansx) , w)
    return rightrate,F

def abnormal(addr):
    s=[]
    for i in range(67):
        s.append(i)
    addr1=ipath+'小区平均用户/小区'+addr
    addre=ipath+'小区参数/小区'+addr
    X = read_csv(addre, usecols=s)  # ,, index_col=s,index_col='time'
    #print(X)
    y =  read_csv(addre,usecols=[67])#,index_col='time'
    # print(X.values)
    #print(y),T1
    X=normal(X)  #数据归一化处理
    dot1,f1 = STL1(addr1,X,y)  #矩阵
    print(dot1)
    addr2=ipath+'小区PDCP流量/小区PDCP'+addr
    X = read_csv(addre, usecols=s)  # ,index_col='time'
    y = read_csv(addre, usecols=[67])  # ,index_col='time'
    X = normal(X)
    dot2,f2 = STL1(addr2,X,y)
    addr3=ipath+'小区平均激活用户数/小区激活'+addr
    X = read_csv(addre, usecols=s)  # ,index_col='time'
    y = read_csv(addre, usecols=[67])  # ,index_col='time'
    X = normal(X)
    dot3,f3 = STL1(addr3,X,y)
    path = ipath+ str("预测异常点/" + addr)  # 同目录下的新文件名
    print(dot1)
    with open(path, 'w', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        headers = ['未来一天内小区平均用户的异常情况','未来一天内小区PDCP流量的异常情况','未来一天内小区平均激活用户数的异常情况']
        csv_write.writerow(headers)
        for i in range(len(dot2)):
            row=[dot1[i],dot2[i],dot3[i]]
            csv_write.writerow(row)
    return [addr,f1,f2,f3]


def main():
    rowss=[]
    for i in range(1,59):
        a=26019000+i
        add=str("%s.csv" % (a))
        F=abnormal(add)
        rowss.append(F)
    with open(ipath+"预测异常点/预测水平F1.csv", 'w', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        headers = ['小区编号','未来一天内小区平均用户的异常情况','未来一天内小区PDCP流量的异常情况','未来一天内小区平均激活用户数的异常情况']
        csv_write.writerow(headers)
        for i in range(len(rowss)):
            csv_write.writerow(rowss[i])

main()




