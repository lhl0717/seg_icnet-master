#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
# author:"Zhang Shuyu"
from sys import argv

'''单个图像转为序列化的数据并存储为str供导入json'''

from base64 import b64encode
from json import dumps
import os

class img_to_json(object):
    """
        这个类是用来将图像数据转化成json文件的，方便下一步的处理。主要是为了获取
        图像的字符串信息
    """
    def __init__(self, process_img_path='/media/tyy/learning/YOLACT/yolact_data/coco/images/train/',
                 img_name = '',
                 img_type=".jpg",
                 # out_file_path="D:/firefoxDL/ICNet-tensorflow-master/data/Original_pic_json",
                 out_file_type=".json"):
        """
        :param process_img_path: 待处理图片的路径
        :param img_type: 待处理图片的类型
        :param out_file_path: 输出文件的路径
        :param out_file_type: 输出文件的类型
        使用glob从指定路径中获取所有的img_type的图片
        """
        # self.process_img = glob.glob(process_img_path + "/027496" + img_type)
        self.process_img = process_img_path + img_name
        # self.out_file = out_file_path
        self.out_file_type = out_file_type
        self.img_type = img_type
        self.img_name = img_name

    def en_decode(self):
        """
        对获取的图像数据进行编码，解码后并存储到指定文件，保存为json文件
        :return: null
        """
        count = 0
        print('-' * 30)
        print("运行 Encode--->Decode\nStart process.....\nPlease wait a moment")
        print('-' * 30)
        """
        Start process.....   Please wait a moment
        """
        """filepath, shotname, extension, tempfilename:目标文件所在路径，文件名，文件后缀,文件名+文件后缀"""
        def capture_file_info(filename):
            (filepath, tempfilename) = os.path.split(filename)
            (shotname, extension) = os.path.splitext(tempfilename)
            return filepath, shotname, extension, tempfilename

        ENCODING = 'utf-8'  # 编码形式为utf-8

        # SCRIPT_NAME, IMAGE_NAME, JSON_NAME = argv  # 获得文件名参数

        img = self.process_img  # 所有图片的形成的列表信息
        # img_number = capture_file_info(img)[1]
        # imgs = sorted(img,key=lambda )

        # out_file_path = self.out_file

        # imgtype = self.img_type

        out_file_type = self.out_file_type
        print("待处理的图片:",self.process_img)
        # if len(img) == 0:
        if not self.img_name:
            print("There was nothing under the specified path.")
            return 0
        # for imgname in img:
        else:
            # midname = imgname[imgname.rindex("\\"):imgname.rindex("." + imgtype)]
            midname = capture_file_info(img)[1]   # midname:图片的名称，不带后缀名
            IMAGE_NAME = img
            # IMAGE_NAME = midname + imgtype
            JSON_NAME = midname + out_file_type
            # 读取二进制图片，获得原始字节码，注意 'rb'
            with open(IMAGE_NAME, 'rb') as jpg_file:
                byte_content = jpg_file.read()
            # 把原始字节码编码成 base64 字节码
            base64_bytes = b64encode(byte_content)
            # 将 base64 字节码解码成 utf-8 格式的字符串
            base64_string = base64_bytes.decode(ENCODING)
            # 用字典的形式保存数据
            """raw_data:用来存放加入特性的数据，img_raw_data:用来存放不加入特性的数据，只有图片的字符串数据"""
            # raw_data = {}
            # raw_data["name"] = IMAGE_NAME
            # raw_data["image_base64_string"] = base64_string
            img_raw_data = {}
            img_raw_data = base64_string
            # 将字典变成 json 格式，indent =2:表示缩进为 2 个空格
            # json_data = dumps(raw_data)
            json_img_data = img_raw_data
            return json_img_data
            # 将 json 格式的数据保存到指定的文件中
            # with open(out_file_path+JSON_NAME, 'w') as json_file:
            #     json_file.write(json_img_data)




if __name__ =="__main__":
    trans = img_to_json(img_name='027497.jpg')
    trans.en_decode()