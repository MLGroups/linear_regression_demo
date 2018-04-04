# -*- coding: utf-8 -*-
# @Time     :2018/4/4 下午3:39
# @Author   :李二狗
# @Site     :
# @File     :demo.py
# @Software :PyCharm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
import time

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

path = 'brain_body.txt'

# 加载数据
df = pd.read_fwf(path)

# 异常数据处理

#特征值选择

X = df.iloc[:, 0:1]
Y = df.iloc[:, 1:2]

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 数据模型建立
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型训练
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, Y_train)

# 测试模型预测
Y_predict = lr.predict(X_test)

# 输出模型训练得到的相关参数
# 注意：第1、2、6个系数为0
print("模型的系数(θ):", end="")
print(lr.coef_)
print("模型的截距:", end='')
print(lr.intercept_)

print("训练集上R^2:", lr.score(X_train, Y_train))
print("测试集上R^2:", lr.score(X_test, Y_test))


# 预测值和实际值画图比较
t = np.arange(len(X_test))
# 建一个画布，facecolor是背景色
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, Y_predict, 'g-', linewidth=2, label='预测值')
plt.plot(X_test, Y_predict)
plt.scatter(X_train, Y_train)
# 显示图例，设置图例的位置
plt.legend(loc = 'upper left')
plt.title("线性回归预测身体和脑重之间的关系", fontsize=20)
# 加网格
plt.grid(b=True)
# 保存图片
plt.savefig('demo.png', dpi=200)
plt.show()