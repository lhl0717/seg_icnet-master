import os
import numpy as np
import cv2 as cv
import json
import skimage
import time
import collections
import pickle as pk
from collections import OrderedDict
import labelme

'''
！!！!！Tran VIA to Labelme ！!！!！
'''

# np.set_printoptions(threshold=np.nan)


# We mostly care about the x and y coordinates of each region

def json_reader(json_path):
    json_content = ''
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_content = json_content + line
            # print(line)
    return json_content

def tran_Labelme(json_path):
    annotations = json.loads(json_reader(json_path), object_pairs_hook=OrderedDict)  # 加载json文件
    # json文件中字典嵌套字典，字典嵌套列表。为了取到关键的x，y点的数据，得一一层把字典，列表剥开。

    annotations = list(annotations.values())
    # time.sleep(1)
    print("annotations", annotations)

    # full data:
    # annotations
    # [OrderedDict([
    # ('ui', OrderedDict([('annotation_editor_height', 25),
    # ('annotation_editor_fontsize', 0.8),
    # ('leftsidebar_width', 18),
    # ('image_grid', OrderedDict([('img_height', 80),
    # ('rshape_fill', 'none'),
    # ('rshape_fill_opacity', 0.3),
    # ('rshape_stroke', 'yellow'),
    # ('rshape_stroke_width', 2),
    # ('show_region_shape', True),
    # ('show_image_policy', 'all')])),
    # ('image', OrderedDict([('region_label', 'region_id'),
    # ('region_label_font', '10px Sans')]))])),
    # ('core', OrderedDict([('buffer_size', '18'),
    # ('filepath', OrderedDict()),
    # ('default_filepath', '/9/')])),
    # ('project', OrderedDict([('name', '9')]))]),

    # OrderedDict([('027496.jpg1953736',
    # OrderedDict([('filename', '027496.jpg'),
    # ('size', 1953736),
    # ('regions',
    # [OrderedDict([('shape_attributes', OrderedDict([('name', 'polygon'),
    # ('all_points_x', [1075, 1191, 1269, 1234, 1140, 1143]),
    # ('all_points_y', [931, 911, 1413, 1415, 1403, 1405])])),
    # ('region_attributes', OrderedDict([('antenna', 'antenna')]))]),
    #
    # OrderedDict([('shape_attributes', OrderedDict([('name', 'polygon'),
    # ('all_points_x', [1905, 1993, 2039, 2074, 2023, 1925]),
    # ('all_points_y', [537, 525, 525, 1193, 1209, 1188])])),
    # ('region_attributes', OrderedDict([('antenna', 'antenna')]))]),
    #
    # OrderedDict([('shape_attributes', OrderedDict([('name', 'polygon'),
    # ('all_points_x', [2435, 2478, 2442, 2399]),
    # ('all_points_y', [530, 540, 1027, 1019])])),
    # ('region_attributes', OrderedDict([('antenna', 'antenna')]))]),
    #
    # OrderedDict([('shape_attributes', OrderedDict([('name', 'polygon'),
    # ('all_points_x', [2657, 2710, 2672, 2624]),
    # ('all_points_y', [711, 717, 1113, 1113])])),
    # ('region_attributes', OrderedDict([('antenna', 'antenna')]))])]),
    # ('file_attributes', OrderedDict())])),
    annotations_point = annotations[1]
    print("annotations_point", annotations_point)

    annotations_point = list(annotations_point.values())
    print("annotations_point2", annotations_point)

    # annotations_point = list(annotations_point.values())
    # print("annotations_point2", annotations_point)

    annotations_point = [a for a in annotations_point if a['regions']]
    print("annotations_point3", annotations_point[0]['regions'])

    for i in range(len(annotations_point)):
        print(annotations_point[i]['filename'])

    return annotations_point





