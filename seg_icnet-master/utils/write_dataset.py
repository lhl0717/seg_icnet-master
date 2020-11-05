import os
from PIL import Image
from shutil import copyfile

def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)

# pic_file = 'D:/9/data'
# pic_file = 'D:/demo/data2'
pic_file = 'F:/ICNet_dataset'

b = os.listdir(pic_file)
train_count = len(b)*10/10
f = open('D:/demo/seg_icnet-master/list/antenna_train.txt', 'w+')
# f2 = open('D:/demo/seg_icnet-master/seg_icnet-master/list/antenna_val2.txt', 'w+')
i = 0
for dir in b:
    # i+=1
    # if int(dir.split('_')[0]) > 0:
    img = dir + '/img.png'
    # print(img)
    # produceImage(img, 1280, 720, img)

    label = dir + '/label.png'
    # produceImage(label, 2048, 1024, label)

        # src_img = pic_file + '/' + dir + '/img.png'
        # des_img = '../input/val_antenna/%04d.png' % i
        # copyfile(src_img, des_img)

    line = img + ' ' + label + '\n'
    print(line)
    if b.index(dir) < train_count:
        f.writelines(line)
#         else:
#             f2.writelines(line)

f.close()






