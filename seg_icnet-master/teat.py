import matplotlib
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
# A = np.array([1,2,3,4,5,6,7])
# B = np.array([2,4,6,8])
# mask = np.in1d(A, B)
# print(np.where(mask)[0])
# print(np.where(~mask)[0])
# # print(timeit.timeit(np.where(np.in1d(A, B))[0]))
# # %timeit np.where(pd.Index(pd.unique(B)).get_indexer(A) >= 0)[0]
# print(np.where(pd.Index(pd.unique(B)).get_indexer(A) >= 0)[0])
# a = np.array([685, 379])
# b = np.array([4, 1])
# c = np.array([3, 4])
# d = np.array([4, 5])
# row = []
# row = [[c,b],[a,b],[d,c]]
# # row.append([a,b])
# # row.clear()
# ac = np.abs(np.array(row) - [a,b])
# print(row)
# print(ac)
# print(np.sum(ac, axis=1))
# print(np.sum(np.sum(ac, axis=1), axis=1))
# dis = np.argmin(np.sum(np.sum(ac, axis=1), axis=1))
# print(dis)
# print(row.pop(dis))
# print(row)
# print(ac)
# row.clear()
# print(row)


# a = np.array([[1011  ,347],
#  [1012  ,347],
#  [1013  ,347],
#  [1014  ,346],
#  [1015  ,346],
#  [1016  ,345],
#  [1017  ,342],
#  [1018  ,339]])
# print(a)
# print(a[:, 1:].flatten())
# print(a[:, :1].flatten())
# x = a[:, :1].flatten()
# y = a[:, 1:].flatten()
# z1 = np.polyfit(x, y, 1)  #一次多项式拟合，相当于线性拟合
# p1 = np.poly1d(z1)
# y1 = p1(x)
# print(z1)
# print(p1)
# print(y1)
#
# plt.figure()
# plt.scatter(x, y, 25,"red")
# plt.plot(x, y1, 'blue')
# plt.show()
# zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf")
# myfont = matplotlib.font_manager.FontProperties(fname="D:/汉仪大宋简.ttf")
# ic = [0.33087407, 0.37874500000000033, 0.3217659999999998, 0.37129429999999974, 0.3101232000000005, 0.31429240000000025, 0.31426970000000054, 0.3135724, 0.3262166999999998, 0.36229650000000024, 0.44143120000000025, 0.38229120000000005, 0.3849980999999998, 0.3778145999999998, 0.38419000000000114, 0.3794529000000004, 0.4133988999999989]
# mask = [1.225706199999999, 1.0174294999999995, 1.0147604000000001, 1.0483799999999999, 1.0540491000000003, 1.0336472000000008, 1.0425742000000007, 1.0241954999999994, 1.0425874000000004, 1.0621910999999997, 1.062658599999999, 1.2348272999999992, 1.1889198000000007, 1.0930329000000008, 1.1317442, 1.1593012000000016, 0.9901477000000014]
# x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# ic_deg = [6.25,5.91,5.79,6.3,6.24,6.24,6.09,6.13,6.42,6.5,5.8,5.27,6.37,5.97,6.42,6.52,5.74]
# mask_deg = [5.57,5.86,5.51,5.17,6.12,6.77,5.43,5.47,6.12,5.86,5.0,5.0,5.48,5.27,5.48,5.6,5.14]
# print(len(x), len(ic))
# plt.title('1280*720分辨率Mask RCNN与ICNet对6°样本检测结果对比',  fontproperties=myfont)
# plt.plot(ic_deg, 'r', label = 'ICNet')
# plt.plot(ic_deg, 'or')
# plt.plot(mask_deg, 'g', label = 'Mask RCNN')
# plt.plot(mask_deg, 'og')
# plt.grid(True)
# plt.xlabel("图片",  fontproperties=myfont)
# plt.ylabel("检测角度(°)",  fontproperties=myfont)
# plt.xticks(x)
# plt.legend()
# plt.show()


# if __name__ == '__main__':
#     import torch
#     print(torch.cuda.is_available())


import cv2
import threading
import time


class Producer(threading.Thread):
    """docstring for Producer"""

    def __init__(self, rtmp_str):

        super(Producer, self).__init__()

        self.rtmp_str = rtmp_str

        # 通过cv2中的类获取视频流操作对象cap
        self.cap = cv2.VideoCapture(self.rtmp_str)

        # 调用cv2方法获取cap的视频帧（帧：每秒多少张图片）
        # fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(self.fps)

        # 获取cap视频流的每帧大小
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)
        print(self.size)

        # 定义编码格式mpge-4
        self.fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

        # 定义视频文件输入对象
        self.outVideo = cv2.VideoWriter('saveDir1.avi', self.fourcc, self.fps, self.size)

    def run(self):

        print('in producer')

        ret, image = self.cap.read()

        while ret:
            # if ret == True:

            self.outVideo.write(image)

            cv2.imshow('video', image)

            cv2.waitKey(int(1000 / int(self.fps)))  # 延迟

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.outVideo.release()

                self.cap.release()

                cv2.destroyAllWindows()

                break

            ret, image = self.cap.read()


if __name__ == '__main__':
    print('run program')
    # rtmp_str = 'rtmp://live.hkstv.hk.lxdns.com/live/hks'  # 经测试，已不能用。可以尝试下面两个。
    # rtmp_str = 'rtmp://media3.scctv.net/live/scctv_800'  # CCTVrtmp://58.200.131.2:1935/livetv/hunantv
    rtmp_str = 'rtmp://192.168.31.46:1935/live/abc'  # 湖南卫视

    producer = Producer(rtmp_str)  # 开个线程
    producer.start()
