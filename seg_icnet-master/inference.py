from __future__ import print_function

import argparse
import math
import os
import glob
import sys
import timeit
from tqdm import trange
import tensorflow as tf
import numpy as np
from scipy import misc

from antenna_inference import Antenna, same_atn, same_point, same_row, atn_2_row, init_atns
from model import ICNet, ICNet_BN
from tools import decode_labels
import time
import cv2


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
# define setting & model configuration
ADE20k_class = 150  # predict: [0~149] corresponding to label [1~150], ignore class 0 (background)
cityscapes_class = 19
antenna_classes = 2

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy',
               'trainval': './model/icnet_cityscapes_trainval_90k.npy',
               'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
               'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
               'others': './snapshots/model.ckpt-5000'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

snapshot_dir = './snapshots/'
SAVE_DIR = './output/'


class Segment(object):

    def __init__(self, model_name, model_path, num_classes, input_shape):
        self.model_name = model_name
        self.model_path = model_path
        self.num_classes = num_classes

        self.filter_scale = 1
        self.shape = input_shape

        self.model = None

    def _load_model(self, shape):
        self.x = tf.placeholder(dtype=tf.float32, shape=shape)
        img_tf = self.preprocess(self.x)
        self.img_tf, self.n_shape = self.check_input(img_tf)

        self.model = model_config[self.model_name]
        print(self.model)
        net = self.model({'data': img_tf}, num_classes=self.num_classes,
                         filter_scale=self.filter_scale)
        # net = self.model({'data': img_tf}, num_classes=self.num_classes)

        raw_output = net.layers['conv6_cls']

        # Predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, size=self.n_shape, align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = decode_labels(raw_output_up, shape, self.num_classes)

        # Init tf Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # model_path = model_paths[self.model_name]
        model_path = snapshot_dir
        if self.model_name == 'others':
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                loader = tf.train.Saver(var_list=tf.global_variables())
                load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
                self.load(loader, self.sess, './snapshots/6+20+20/model.ckpt-200000')
            else:
                print('No checkpoint file found.')
        else:
            # model path must be a model
            net.load(model_path, self.sess)
            print('Restore from {}'.format(model_path))

    def seg_img(self, img):
        tic = time.time()
        preds = self.sess.run(self.pred, feed_dict={self.x: img})

        p = np.array(np.where(preds[0] != [90, 90, 90])[:-1])
        p = np.dstack((p[0], p[1]))[0]
        uni_p = np.unique(p, axis=0)
        atn = Antenna()
        atn.l_line.append(uni_p[0])
        atns = []

        for p_i in range(1, len(uni_p)):
            point = uni_p[p_i]
            last_point = uni_p[p_i-1]
            if same_point(point, last_point):
                continue
            if not same_atn(point, last_point):
                atn.r_line.append(last_point)
                # 每次插完右边点检查是否换行
                if not same_row(point, last_point):
                    # 成对插入天线左右边像素点
                    row = atn_2_row(atn)
                    atn.clear()
                    # 第一次换行时初始化天线对象列表atns
                    if not atns:
                        atns = init_atns(row, preds)
                        row.clear()
                        atn.l_line.append(point)
                        continue
                    # 天线左右边像素点组合遍历所有天线对象最后一个点作差，相差最小则插入天线对象
                    for atns_i in range(len(atns)):
                        if not row:
                            break
                        dis = np.abs(np.array(row) - atns[atns_i].last_point())
                        m_dis = np.argmin(np.sum(np.sum(dis, axis=1), axis=1))
                        if np.abs(row[m_dis][0][0] - atns[atns_i].last_point()[0][0]) == 1 \
                                and np.abs(row[m_dis][0][1] - atns[atns_i].last_point()[0][1]) <= 1 \
                                and np.abs(row[m_dis][1][1] - atns[atns_i].last_point()[1][1]) <= 1:
                            atns[atns_i].append(row[m_dis], preds, atns_i)
                            row.pop(m_dis)
                    if row:
                        for row_j in range(0, len(row)):
                            atn_tmp = Antenna()
                            atn_tmp.append(row[row_j], preds, len(atns))
                            atns.append(atn_tmp)
                        row.clear()
                atn.l_line.append(point)

        atns = [atns[atns_i] for atns_i in range(0, len(atns)) if len(atns[atns_i].l_line) > 75]
        for k in range(len(atns)):
            a = np.array(atns[k].l_line)
            x = a[:, :1].flatten()
            dis = int(len(x) / 10)
            x = x[dis:-dis]
            y = a[:, 1:].flatten()[dis:-dis]
            z1 = np.polyfit(x, y, 1)  # 一次多项式拟合，相当于线性拟合
            p1 = np.poly1d(z1)
            y1 = p1(x).astype(np.int)

            b = np.array(atns[k].r_line)
            x2 = b[:, :1].flatten()[dis:-dis]
            y2 = b[:, 1:].flatten()[dis:-dis]
            z2 = np.polyfit(x2, y2, 1)
            p2 = np.poly1d(z2)
            yl2 = p2(x2).astype(np.int)

            cur_img = np.array(img, dtype='float32')
            img = cv2.line(cur_img, (y1[0], x[0]), (y1[-1], x[-1]), (199, 0, 69), 3)
            img = cv2.line(cur_img, (yl2[0], x2[0]), (yl2[-1], x2[-1]), (199, 0, 69), 3)
            l_degree = (math.atan2((x[-1] - x[0]), (y1[-1] - y1[0]))) * 180 / math.pi
            r_degree = (math.atan2((x2[-1] - x2[0]), (yl2[-1] - yl2[0]))) * 180 / math.pi
            degree = round((l_degree + r_degree) / 2 - 90, 2)
            img = cv2.putText(cur_img, str(degree), (yl2[0], x2[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        overlayed_img = cv2.addWeighted(np.array(img, dtype='float32'), 0.4, preds[0], 0.6, 0)
        return overlayed_img, preds

    def seg_video(self, video_f, is_save=True, is_record=True):
        if os.path.exists(video_f):
            save_dir = os.path.join(os.path.dirname(video_f), os.path.basename(video_f).split('.')[0])
            print(save_dir)
            save_record_f = os.path.join(os.path.dirname(video_f), os.path.basename(video_f).split('.')[0] + '.mp4')
            # videoWriter = cv2.VideoWriter('./output/'+ video_f.split('/')[-1].split('.')[0] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (2048, 1024))
            i = 0
            cap = cv2.VideoCapture(video_f)

            while cap.isOpened():
                ret, frame = cap.read()
                tic = time.time()
                if ret:
                    frame_resize = cv2.resize(frame, (2048, 1024),
                                              interpolation=cv2.INTER_AREA)
                    if i == 0:
                        # get the shape from first frame
                        print('Detect input shape once: ', frame_resize.shape)
                        self._load_model(shape=frame_resize.shape)
                    i += 1
                    if i % 2 == 0:
                        pic_name = 'frame_%04d.jpg' % i
                        res, _ = self.seg_img(frame_resize)
                        if is_save:
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            # cv2.imwrite(os.path.join(save_dir, 'frame_%04d.jpg' % i), res)
                            cv2.imwrite(os.path.join(save_dir, pic_name), res)
                        if is_record:
                            # TODO: do some record things
                            # videoWriter.write(res)
                            pass
                        print(pic_name + ' fps: ', round(1 / (time.time() - tic), 4))
                        res1 = res.astype(np.uint8)
                        cv2.imshow('seg', res1)
                        cv2.waitKey(1)
            # videoWriter.release()
            cap.release()
        else:
            print('# video file not exist: '.format(video_f))

    @staticmethod
    def load(saver, sess, ckpt_path):
        saver.restore(sess, ckpt_path)
        print("Restored model parameters from {}".format(ckpt_path))

    @staticmethod
    def load_img(img_path):
        if os.path.isfile(img_path):
            print('successful load img: {0}'.format(img_path))
        else:
            print('not found file: {0}'.format(img_path))
            sys.exit(0)

        filename = img_path.split('/')[-1]
        img = misc.imread(img_path, mode='RGB')
        print('input image shape: ', img.shape)
        return img, filename

    @staticmethod
    def preprocess(img):
        # Convert RGB to BGR
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        img -= IMG_MEAN
        img = tf.expand_dims(img, dim=0)
        return img

    @staticmethod
    def check_input(img):
        ori_h, ori_w = img.get_shape().as_list()[1:3]
        if ori_h % 32 != 0 or ori_w % 32 != 0:
            new_h = (int(ori_h / 32) + 1) * 32
            new_w = (int(ori_w / 32) + 1) * 32
            shape = [new_h, new_w]
            img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
            print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
        else:
            shape = [ori_h, ori_w]

        return img, shape


if __name__ == '__main__':
    # seg = Segment('others', 'model/cityscapes/icnet.npy', cityscapes_class, [1024, 2048])
    seg = Segment('others', './model/model.ckpt-60000', antenna_classes, [1024, 2048])

    # result, _ = seg.seg_img('input/3.png')
    # cv2.imshow('seg', result)
    # cv2.waitKey(1)
    # seg.seg_dir('/media/jintian/sg/permanent/Cityscape/leftImg8bit/demoVideo/stuttgart_00')
    seg.seg_video('./DJI_0419/DJI_0419.mov')
