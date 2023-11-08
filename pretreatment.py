
# 数据预处理

from os import getpid, times
from numpy import double
import pandas as pd

import csv
import codecs
import xlrd
# 工具函数

ipath = "/home/hadoop/Experiment/project/"

# 获取全部的小区编号
# 返回类型：list
def getAllDistrictId():
    filePath = ipath + "核心指标表.xlsx"
    df = pd.read_excel(filePath)
    return list(set(df['小区编号'].values.tolist()))  # numpy.ndarray转list并去重

# print(getAllDistrictId())


# 小区编号列表
districtIdList = [26019001, 26019002, 26019003, 26019004, 26019005, 26019006, 26019007, 26019008,
                  26019009, 26019010, 26019011, 26019012, 26019013, 26019014, 26019015, 26019016, 26019017, 26019018,
                  26019019, 26019020, 26019021, 26019022, 26019023, 26019024, 26019025, 26019026, 26019027, 26019028,
                  26019029, 26019030, 26019031, 26019032, 26019033, 26019034, 26019035, 26019036, 26019037, 26019038,
                  26019039, 26019040, 26019041, 26019042, 26019043, 26019044, 26019045, 26019046, 26019047, 26019048,
                  26019049, 26019050, 26019051, 26019052, 26019053, 26019054, 26019055, 26019056, 26019057, 26019058]


# 根据小区编号获取对应小区的平均用户数列表
# 返回类型：list
def getAverageUser(districtId):
    filePath = ipath +"核心指标表.xlsx"
    df = pd.read_excel(filePath)
    avUserDf = df.loc[df['小区编号'] == districtId, :]
    avUserList = avUserDf['小区内的平均用户数'].values.tolist()
    return avUserList

# print(getAverageUser(26019001))


# 根据小区编号获取对应小区的PDCP流量
# 返回类型：list
def getPDCP(districtId):
    filePath = ipath +"核心指标表.xlsx"
    df = pd.read_excel(filePath)
    pdcpDf = df.loc[df['小区编号'] == districtId, :]
    pdcpList = pdcpDf['小区PDCP流量'].values.tolist()
    return pdcpList

# print(getPDCP(26019001))

# 获取上行：小区PDCP层所接收到的上行数据的总吞吐量比特
def getUpPDCP(districtId):
    filePath = ipath + "核心指标表.xlsx"
    df = pd.read_excel(filePath)
    pdcpDf = df.loc[df['小区编号'] == districtId, :]
    uplist = pdcpDf['小区PDCP层所接收到的上行数据的总吞吐量比特'].values.tolist()
    return uplist

# 获取下行：小区PDCP层所发送的下行数据的总吞吐量比特
def getDownPDCP(districtId):
    filePath = ipath + "核心指标表.xlsx"
    df = pd.read_excel(filePath)
    pdcpDf = df.loc[df['小区编号'] == districtId, :]
    downlist = pdcpDf['小区PDCP层所发送的下行数据的总吞吐量比特'].values.tolist()
    return downlist


# 根据小区编号获取对应小区的平均激活用户数
# 返回类型：list
def getAverageActiveUser(districtId):
    filePath = ipath + "核心指标表.xlsx"
    df = pd.read_excel(filePath)
    avActiveUserDf = df.loc[df['小区编号'] == districtId, :]
    avActiveUserList = avActiveUserDf['平均激活用户数'].values.tolist()
    return avActiveUserList




# 数据写入文件
# 辅助函数
def storFileUtil(data, filePath):
    data = list(map(lambda x: [x], data))
    with open(filePath, 'w', newline='') as f:
        mywrite = csv.writer(f)
        for i in data:
            mywrite.writerow(i)

def writeCoreIntoCsv():
    districtIdList = [26019001, 26019002, 26019003, 26019004, 26019005, 26019006, 26019007, 26019008,
                      26019009, 26019010, 26019011, 26019012, 26019013, 26019014, 26019015, 26019016, 26019017, 26019018,
                      26019019, 26019020, 26019021, 26019022, 26019023, 26019024, 26019025, 26019026, 26019027, 26019028,
                      26019029, 26019030, 26019031, 26019032, 26019033, 26019034, 26019035, 26019036, 26019037, 26019038,
                      26019039, 26019040, 26019041, 26019042, 26019043, 26019044, 26019045, 26019046, 26019047, 26019048,
                      26019049, 26019050, 26019051, 26019052, 26019053, 26019054, 26019055, 26019056, 26019057, 26019058]
    for item in districtIdList:  # 为了命名方便，索引从1开始
        filePath =  ipath +'小区下行/小区下行' + str(item) + '.csv'
        dataList = getDownPDCP(item)
        storFileUtil(dataList, filePath)

writeCoreIntoCsv()


# 已知某小区编号，输入时间点，获取上一个周期的参数资料
def getParameter(districtId, timeString):
    filePath =  ipath +"prodata.xlsx"
    df = pd.read_excel(filePath)
    # df1 = df.loc[df['小区编号'] == districtId & df['时间'], :]
    dic = df.loc[df['小区编号'] == districtId, :].to_dict()
    # dic = df1.loc[df['时间'] == timeString, :].to_dict()
    pList = []
    for innerdic in dic.values():
        for value in innerdic.values():
            pList.append(value)
    # 去除前四个元素，剩下即为需要的参数
    for i in range(4):
        del(pList[0])
    # 列表最后加0，増广
    pList.append(0)
    return pList

# print(getParameter(26019001, '2021-08-28 00:00'))


# 获取全部时间项
# def getAllTime():
#     filePath =  ipath +"test1 - 副本.xlsx"
#     df = pd.read_excel(filePath)
#     return list(set(df['时间'].values.tolist()))  # numpy.ndarray转list并去重

# print(getAllTime())


# 获取某个小区的全部参数
def writeDistrictParameter(districtId):
    filePath =  ipath +"prodata.xlsx"
    df = pd.read_excel(filePath)
    df1 = df.loc[df['小区编号'] == districtId, :]
    df2 = df1.drop(columns=['时间', '基站编号', '小区编号', '本地小区标识'])
    df2['増广'] = 0 # 末尾加一列0
    df2.to_csv( ipath +'小区参数/小区'+str(districtId)+'.csv', header=False, index=False)
    return

# writeDistrictParameter(26019001)

def writeParameterIntoCsv():
    districtIdList = [26019001, 26019002, 26019003, 26019004, 26019005, 26019006, 26019007, 26019008,
                      26019009, 26019010, 26019011, 26019012, 26019013, 26019014, 26019015, 26019016, 26019017, 26019018,
                      26019019, 26019020, 26019021, 26019022, 26019023, 26019024, 26019025, 26019026, 26019027, 26019028,
                      26019029, 26019030, 26019031, 26019032, 26019033, 26019034, 26019035, 26019036, 26019037, 26019038,
                      26019039, 26019040, 26019041, 26019042, 26019043, 26019044, 26019045, 26019046, 26019047, 26019048,
                      26019049, 26019050, 26019051, 26019052, 26019053, 26019054, 26019055, 26019056, 26019057, 26019058]
    for item in districtIdList:  # 为了命名方便，索引从1开始
        writeDistrictParameter(item)

# writeParameterIntoCsv()