# from __future__ import print_function

import argparse
import os
import glob
import sys
import timeit
import pandas as pd

from tqdm import trange
import tensorflow as tf
import numpy as np
from scipy import misc

from model import ICNet, ICNet_BN
from tools import decode_labels,get_labelcolours
import time
import cv2
import math
import matplotlib.pyplot as plt

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
# define setting & model configuration
ADE20k_class = 150  # predict: [0~149] corresponding to label [1~150], ignore class 0 (background)
cityscapes_class = 19
antenna_classes = 2
np.set_printoptions(threshold=np.inf)

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy',
               'trainval': './model/icnet_cityscapes_trainval_90k.npy',
               'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
               'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
               'others': './snapshots/'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

snapshot_dir = './snapshots'
SAVE_DIR = './output/3wdata_test/40w/'
# SAVE_DIR = './output/'


class Args(object):
    def __init__(self, img_path, model, dataset, filter_scale):
        self.img_path = img_path
        self.model = model
        self.dataset = dataset
        self.filter_scale = filter_scale
        self.save_dir = SAVE_DIR
        self.weight_path = ''

class Antenna(object):
    def __init__(self):
        self.l_line = []
        self.r_line = []
    def append(self, point, preds, color_i):
        self.l_line.append(point[0])
        self.r_line.append(point[1])
        preds[0][point[0][0]][point[0][1]] = get_labelcolours(color_i)
        preds[0][point[1][0]][point[1][1]] = get_labelcolours(color_i)
    def clear(self):
        self.l_line.clear()
        self.r_line.clear()
    def last_point(self):
        return [self.l_line[-1], self.r_line[-1]]

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file or input directory.",
                        required=True)
    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=True)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'antenna'],
                        required=True)

    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path, shape):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    img = misc.imread(img_path, mode='RGB')
    if img.shape != shape:
        img = cv2.resize(img, (shape[0], shape[1]))
    print('input image shape: ', img.shape)

    return img, filename

def preprocess(img):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    img = tf.expand_dims(img, dim=0)
    return img

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

def same_row(point, last_point):
    return point[0] == last_point[0]

def near_row(point, last_point):
    return point[0] == last_point[0] + 1

def same_atn(point, last_point):
    return point[1] == (last_point[1] + 1)

def same_point(point, last_point):
    return all(point == last_point)

def atn_2_row(atn):
    row = []
    for atn_i in range(len(atn.l_line)):
        row.append([atn.l_line[atn_i], atn.r_line[atn_i]])
    return row

def init_atns(row, preds):
    atns = []
    for row_i in range(len(row)):
        atn_tmp = Antenna()
        atn_tmp.append(row[row_i], preds, row_i)
        atns.append(atn_tmp)
    return atns


def main():
    global atn
    global preds
    args = Args(
        img_path='D:/val/',
        model='others',
        dataset='antenna',
        filter_scale=1,
    )
    args.weight_path = './snapshots/3wDataSet/model.ckpt-400000'
    # args.weight_path = './snapshots/3wDataSet/model.ckpt-300000'


    # args = get_arguments()

    if args.dataset == 'cityscapes':
        num_classes = cityscapes_class
    elif args.dataset == 'antenna':
        num_classes = antenna_classes
    else:
        num_classes = ADE20k_class

    # Read images from directory (size must be the same) or single input file
    imgs = []
    ori_imgs = []
    filenames = []
    # [2048, 1024],[1280, 720],[3840,2160]
    load_shape = [2048, 1024]
    if os.path.isdir(args.img_path):
        file_paths = glob.glob(os.path.join(args.img_path, '*'))
        for file_path in file_paths:
            pic_files = glob.glob(os.path.join(file_path, '*'))
            for pic_file in  pic_files:
                ext = pic_file.split('jpg')
                if len(ext) > 2:
                    continue
                img, filename = load_img(pic_file, load_shape)
                imgs.append(img)
                ori_imgs.append(img)
                filenames.append(filename.replace('\\', '/'))
        print(filenames)
    else:
        img, filename = load_img(args.img_path)
        imgs.append(img)
        filenames.append(filename)

    shape = imgs[0].shape[0:2]

    x = tf.placeholder(dtype=tf.float32, shape=img.shape)
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)
    model = model_config[args.model]
    net = model({'data': img_tf}, num_classes=num_classes, filter_scale=args.filter_scale)
    raw_output = net.layers['conv6_cls']

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = decode_labels(raw_output_up, shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()

    model_path = model_paths[args.model]
    if args.model == 'others':
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=tf.global_variables())
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            # load(loader, sess, ckpt.model_checkpoint_path)
            load(loader, sess, args.weight_path)

        else:
            print('No checkpoint file found.')
    else:
        net.load(model_path, sess)
        print('Restore from {}'.format(model_path))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    val_time = []
    det_dict = {1: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 12: 0, 15: 0}
    fit_dict = {1: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 12: 0, 15: 0}
    for i in trange(len(imgs), desc='Inference', leave=True):
        # print(filenames[i].split('\\'))
        act_deg = int(filenames[i].split('/')[1])
        fit_count = 0
        det_count = 0
        start_time = timeit.default_timer()
        preds = sess.run(pred, feed_dict={x: imgs[i]})
        os.makedirs(args.save_dir + filenames[i].replace(filenames[i].split("/")[-1], ''), exist_ok=True)
        misc.imsave(args.save_dir + filenames[i].split('.')[0] + '_Pred.jpg', preds[0])
        # misc.imsave(args.save_dir + filenames[i].split('.')[0] + 'Ori.jpg', ori_imgs[i])


        # print(preds[0][0][0])
        p = np.array(np.where(preds[0] != [90, 90, 90])[:-1])
        p = np.dstack((p[0], p[1]))[0]

        if p.shape[0] == 0:
            continue
        uni_p = np.unique(p, axis=0)
        atns = []
        atn = Antenna()
        atn.l_line.append(uni_p[0])
        # 遍历所有标记为天线类别的像素点
        for p_i in range(1, len(uni_p)):
            point = uni_p[p_i]
            last_point = uni_p[p_i-1]
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
                                and np.abs(row[m_dis][0][1] - atns[atns_i].last_point()[0][1]) <= 1\
                                and np.abs(row[m_dis][1][1] - atns[atns_i].last_point()[1][1]) <= 1:
                            # print(np.abs(row[m_dis][0][1] - atns[atns_i].last_point()[0][1]), np.abs(row[m_dis][1][1] - atns[atns_i].last_point()[1][1]))
                            atns[atns_i].append(row[m_dis], preds, atns_i)
                            row.pop(m_dis)
                    if row:
                        for row_j in range(0, len(row)):
                            atn_tmp = Antenna()
                            atn_tmp.append(row[row_j], preds, len(atns))
                            atns.append(atn_tmp)
                        row.clear()
                atn.l_line.append(point)

        # 遍历天线对象，去除左右边拟合计算下倾角
        atns = [atns[atns_i] for atns_i in range(0, len(atns)) if len(atns[atns_i].l_line)>75]
        for k in range(len(atns)):
            a = np.array(atns[k].l_line)
            # 天线左边
            x1 = a[:, :1].flatten()
            dis = int(len(x1)/10)
            x1 = x1[dis:-dis]
            y = a[:, 1:].flatten()[dis:-dis]
            z1 = np.polyfit(x1, y, 1)  # 一次多项式拟合，相当于线性拟合
            p1 = np.poly1d(z1)
            y1 = p1(x1).astype(np.int)

            # 天线右边
            b = np.array(atns[k].r_line)
            x2 = b[:, :1].flatten()[dis:-dis]
            y2 = b[:, 1:].flatten()[dis:-dis]
            z2 = np.polyfit(x2, y2, 1)
            p2 = np.poly1d(z2)
            yl2 = p2(x2).astype(np.int)

            # plt.figure()
            # plt.scatter(x1, y, 25, "red")
            # plt.scatter(x2, y2, 25, "green")
            #
            # plt.plot(x1, y1, 'blue')
            # plt.plot(x2, yl2, 'yellow')

            # plt.show()

            cur_img = np.array(ori_imgs[i], dtype='float32')
            ori_imgs[i] = cv2.line(cur_img, (y1[0], x1[0]), (y1[-1], x1[-1]), (199, 0, 69), 3)
            ori_imgs[i] = cv2.line(cur_img, (yl2[0], x2[0]), (yl2[-1], x2[-1]), (199, 0, 69), 3)
            # l_degree = (x1[-1] - x1[0])/(y1[-1] - y1[0])
            l_degree = (math.atan2((x1[-1] - x1[0]), (y1[-1] - y1[0])))*180/math.pi
            # r_degree = (x2[-1] - x2[0])/(yl2[-1] - yl2[0])
            r_degree = (math.atan2((x2[-1] - x2[0]), (yl2[-1] - yl2[0])))*180/math.pi
            degree = abs(round((l_degree+r_degree)/2-90, 2))
            acc_dis = abs(degree - act_deg)
            if acc_dis < 3:
                det_count = det_count + 1
            if acc_dis < 1:
                fit_count = fit_count + 1
            # print(l_degree,r_degree, degree)
            ori_imgs[i] = cv2.putText(cur_img, str(degree), (yl2[0], x2[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            # print((y1[0], x[0]), (y1[-1], x[-1]))
            # print((yl2[0], x2[0]), (yl2[-1], x2[-1]))

        overlayed_img = cv2.addWeighted(np.array(ori_imgs[i], dtype='float32'), 0.7, preds[0], 0.3, 0)
        # os.makedirs(args.save_dir + filenames[i].split('Screen')[0], exist_ok=True)
        fit_dict[act_deg] += fit_count
        det_dict[act_deg] += det_count
        # print(args.save_dir + filenames[i], det_count, fit_count)
        misc.imsave(args.save_dir + filenames[i], overlayed_img)

        print(args.save_dir + filenames[i])
        elapsed = timeit.default_timer() - start_time
        print('inference time: {}'.format(elapsed))
        val_time.append(elapsed)
    print('mean_val_time:', np.mean(val_time[1:]))
    print('detection:', det_dict, '%.3f' % (sum(det_dict.values())/158*100) + "%")
    print('fitting:', fit_dict, '%.3f' % (sum(fit_dict.values())/158*100) + "%")
    print(val_time)




if __name__ == '__main__':

    main()
