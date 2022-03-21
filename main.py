import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from math import pi
import time

# ===============================贪心法求解模块=================================
def tanXin(m, h, v):
    data = open("C:/Users/涵涵/Desktop/checkresult.txt", "w+")
    start = time.time()
    arr = [(i, v[i] / h[i], h[i], v[i]) for i in range(len(h))]
    arr.sort(key = lambda x: x[1], reverse = True)
    bagVal = 0
    bagList = [0] * len(h)
    for i, w, h, v in arr:
        if w <= m:
            m -= h
            bagVal += v
            bagList[i] = 1
        else:
            bagVal += m * w
            bagList[i] = 1
            break

    end = time.time()
    print('最大价值:', bagVal)
    print('最大价值:', bagVal, file=data)
    print('解向量:', bagList)
    print('解向量:', bagList, file=data)
    return bagVal

def bag(n, m, w, v):
    value = [[0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if j < w[i - 1]:
                value[i][j] = value[i - 1][j]
            else:
                value[i][j] = max(value[i - 1][j], value[i - 1][j - w[i - 1]] + v[i - 1])   # 背包总容量够放当前物体，取最大价值
    return value

# ===============================动态规划求解模块=================================
def DP(n, m, w, value):
    data = open("C:/Users/涵涵/Desktop/checkresult.txt", "w+")
    bagList = [0] * len(w)
    x = [0 for i in range(n)]
    j = m
    # 求出解向量
    for i in range(n, 0, -1):
        if value[i][j] > value[i - 1][j]:
            x[i - 1] = 1
            j -= w[i - 1]
    for i in range(n):
        if x[i]:
            bagList[i] = 1
    print('最大价值为:', value[n][m])
    print('最大价值为:', value[n][m], file=data)
    print("解向量:", bagList)
    print('解向量:', bagList, file=data)
    return value[n][m]

bestV = 0
curW = 0
curV = 0
bestx = None

# ============================回溯求解模块=================================
def DFS(i, n, m, w, v):
    global bestV, curW, curV, x, bestx
    if i >= n:
        if bestV < curV:
            bestV = curV
            bestx = x[:]
    else:
        if curW + w[i] <= m:
            x[i] = 1
            curW += w[i]
            curV += v[i]
            DFS(i + 1, n, m, w, v)
            curW -= w[i]
            curV -= v[i]
        x[i] = 0
        DFS(i + 1, n, m, w, v)

# 绘制散点图
def show(w, v):
    plt.title("value-weight", fontsize = 20)   # 图名称
    plt.xlabel("weight", fontsize = 10)
    plt.ylabel("value", fontsize = 10)
    plt.axis([0, 100, 0, 100])   # 设置x,y轴长度
    plt.scatter(w, v, s = 20)   # 将数据传入x,y轴
    plt.show()

# 绘制价值/重量折线图
def show_zx():
    number = [1, 2, 3, 4, 5, 6]
    rate=[0.05,1.194,1,3,0.92,0.909]
    #绘制图形
    # 参数linewidth设置plot()绘制的线条的粗细
    plt.plot(number,rate, linewidth=5)
    #语法：plot(x轴坐标，y轴坐标，其他参数设置)
    # 设置图表标题，设置字体大小
    #函数title()给图表指定标题，参数fontsize指定了图表中文字的大小。
    plt.title("value/weight", fontsize=24)
    #给x轴添加标签，设置字体大小
    plt.xlabel("Number", fontsize=14)
    # 给y轴添加标签，设置字体大小
    plt.ylabel("value/weight", fontsize=14)
    # 设置每个坐标轴的取值范围
    plt.axis([0, 10, 0, 3])   #[x.x,x.y,y.x,y.y]
    # tick_params()设置刻度标记的大小，设置刻度的样式
    plt.tick_params(axis='both', labelsize=14)
    # 打开matplotlib查看器，并显示绘制的图形
    plt.show()


# 排序函数
def sort(w, v):
    w1 = np.array(w)
    v1 = np.array(v)
    vw = v1 * (1 / w1)
    vw = abs(np.sort(-vw))
    print("价值/重量比递减排序: ")
    for x in vw:
        print('%.4f' % x, end=' ')

# 读取数据
def get_Data():
    fileName = str(input('请输入文件名'))
    a = np.loadtxt("C:/Users/涵涵/Desktop/测试数据/" + fileName)
    return a

# ========================主函数=======================
if __name__ == '__main__':
    a = get_Data()
    a = a.ravel()
    m = int(a[0])  # 背包容量
    n = int(a[1])  # 物品个数
    h = 2 * n + 1
    print(a[2:h])
    i = 2
    w = []  # 物品重量
    v = []  # 物品价值
    while i < 2 * n + 2:
        if i % 2 == 0:
            w.append(a[i])
        else:
            v.append(a[i])
        i = i + 1
    w = list(map(int, w))
    v = list(map(int, v))
    s = int(input('请选择算法\n1、贪心法\n2、回溯法\n3、动态规划法\n'))
    if s == 1:
        start = time.time()
        tanXin(m, w, v)
        end = time.time()
        data = open("C:/Users/涵涵/Desktop/checkresult.txt", "a+")
        print("运行时间", end - start, "s", file=data)
        data.close()
        print("运行时间", end - start, "s")
    elif s == 2:
        data = open("C:/Users/涵涵/Desktop/checkresult.txt", "w+")
        start = time.time()
        x = [0 for i in range(n)]
        DFS(0, n, m, w, v)
        # bestV = float(bestV)
        print("最大价值:", bestV)
        print("最大价值:", bestV, file=data)
        print("解向量:", bestx)
        print("解向量:", bestx, file=data)
        end = time.time()
        print("运行时间", end - start, "s", file=data)
        print("运行时间:", end - start, "s")
        data.close()
    elif s == 3:
        start = time.time()
        value = bag(n, m, w, v)
        DP(n, m, w, value)
        end = time.time()
        data = open("C:/Users/涵涵/Desktop/checkresult.txt", "a+")
        print("运行时间", end - start, "s", file=data)
        data.close()
        print("运行时间:", end - start, "s")
    else:
        print("输入错误！")

    sort(w, v)
    show(w, v)
    show_zx()