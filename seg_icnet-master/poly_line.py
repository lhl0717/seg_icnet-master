import matplotlib.pyplot as plt
import numpy as np
import math

def poly_line(point_list):
    line = np.array(point_list)
    # 提取x，y轴坐标
    x_list = line[:, :1].flatten()
    y_list = line[:, 1:].flatten()
    # 去除上下部分，上下方去除长度个占总长度十分之一
    dis = int(len(x_list) / 10)
    x_list = x_list[dis:-dis]
    y_list = y_list[dis:-dis]
    # 求拟合方程，代入x得拟合直线
    z = np.polyfit(x_list, y_list, 1)
    p = np.poly1d(z)
    y1 = p(x_list).astype(np.int)
    # 返回x，y和拟合曲线y1坐标集合
    return x_list, y_list, y1

# 两点斜率计算弧度输出角度
def cal_deg(x, y):

    return (y[-1] - y[0])/(x[-1] - x[0]), (math.atan2((y[-1] - y[0]), (x[-1] - x[0]))) * 180 / math.pi

# 拟合左右边曲线，返回左右边角度平均值
def polyLR(l_line, r_line):

    l_x, l_y, l_y1 = poly_line(l_line)
    r_x, r_y, r_y1 = poly_line(r_line)

    # plt.figure()
    # 绘制左右边散点曲线和拟合直线
   # plt.scatter(l_y, l_x, 15, 'red')
    plt.plot(l_y1, l_x, 15, 'blue')
   # plt.scatter(r_y, r_x, 15, 'red')
    plt.plot(r_y1, r_x, 15, 'yellow')

    print(l_x[0], l_y1[0], l_x[-1], l_y1[-1])
    print(r_x[0], r_y1[0], r_x[-1], r_y1[-1])
    # print(r_x, r_y1)
    # 计算左右边角度
    l_rate, l_deg = cal_deg(l_x, l_y1)
    r_rate, r_deg = cal_deg(r_x, r_y1)
    # 取平均值
    avg_deg = round((l_deg + r_deg)/2, 2)

    print(l_deg , r_deg, avg_deg)

    return avg_deg


