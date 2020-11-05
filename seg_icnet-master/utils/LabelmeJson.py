import json
from collections import OrderedDict
import numpy as np
import os
import pickle as pk
from utils.VIAjson import *
from utils.img2jsonV20 import img_to_json

IMAGE_DIR = 'D:/train/2/'
json_path = 'D:/train/第三次json文件/2.json'

Vannotations_point = tran_Labelme(json_path)

dir_list = os.listdir(IMAGE_DIR)

pic_list = list(filter(lambda x: x.split('.')[1] in '.jpg', dir_list))

print(Vannotations_point)

# print(Vannotations_point[0]['regions'][0]['shape_attributes']['all_points_x'])
# print(Vannotations_point[0]['regions'][0]['shape_attributes']['all_points_y'])

fit = []
unfit = []
p = int(pic_list[0].split('.')[0])
for i in range(len(Vannotations_point)):
    t = int(Vannotations_point[i]['filename'].split('.')[0])
    if p != t:
        unfit.append(p)
        p = t
        p += 1
        fit.append(t)
        continue
    fit.append(p)
    p += 1
print(fit,'\n',unfit)

read_path = '002615.json'
Lannotations = json.loads(json_reader(read_path), object_pairs_hook=OrderedDict)  # 加载json文件

for v_index in range(9, len(Vannotations_point)):   # 遍历VIA_json图片
# for v_index in range(0,5):
    filename = Vannotations_point[v_index]['filename']
    print('\n',filename)
    if int(filename.split('.')[0]) in fit:
        trans = img_to_json(img_name = filename, process_img_path=IMAGE_DIR)
        Lannotations['imagePath'] = filename  # 写入图片名
        Lannotations['imageData'] = trans.en_decode()  # 写入图片数据
        save_path = IMAGE_DIR + filename.split('.')[0] +'.json'
        regions = Vannotations_point[v_index]['regions']

        dis = len(Lannotations['shapes']) - len(regions)
        print(len(Lannotations['shapes']), len(regions))
        if dis < 0:
            for i in range(0, abs(dis)):
                null_shape = OrderedDict(Lannotations['shapes'][0])
                null_shape['points'] = []
                Lannotations['shapes'].append(null_shape)
        elif dis >0:
            Lannotations['shapes'] = Lannotations['shapes'][:-dis]
        print('len_compare:',len(Lannotations['shapes']),'*'*5,len(regions))
        print(Lannotations['shapes'])
        for shape_index in range(0, len(regions)):  # 遍历图片内的regions
            points = []
            print(shape_index)
            Lannotations['shapes'][shape_index]['points'] = []
            x_list = regions[shape_index]['shape_attributes']['all_points_x']
            y_list = regions[shape_index]['shape_attributes']['all_points_y']
            for xy_index in range(0, len(x_list)):  # 遍历regions的点坐标列表
                point = []
                point = [x_list[xy_index], y_list[xy_index]]
                print(point)
                points.append(point)
            Lannotations['shapes'][shape_index]['points'] = points# 写入坐标信息
            print(Lannotations['shapes'][shape_index]['points'])
        print('regions:', regions)
        print('shapes:' ,Lannotations['shapes'])
        print('-'*10)
        print(Lannotations['shapes'])
        json_data = json.dumps(Lannotations)
        with open(save_path, 'w') as json_file:
            # json.dump(Lannotations, json_file)
            json_file.write(json_data)