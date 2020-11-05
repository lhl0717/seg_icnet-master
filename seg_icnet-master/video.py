import os
import shutil

import cv2

def video2img(videoPath,imgPath):
    vc = cv2.VideoCapture(videoPath)  # 读入视频文件
    c = 0
    d = 0
    rval = vc.isOpened()
    # timeF = 1  #视频帧计数间隔频率
    while rval:  # 循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        #    if(c%timeF == 0): #每隔timeF帧进行存储操作
        #        cv2.imwrite('smallVideo/smallVideo'+str(c) + '.jpg', frame) #存储为图像
        if rval:
            # img为当前目录下新建的文件夹
            if (c % 2 == 0):  # 更改帧数： 1：不跳过，30帧    2：跳过偶数，15帧
                d = d + 1
                if d > 2:
                    break
                frame_resize = cv2.resize(frame, (720, 480),
                                          interpolation=cv2.INTER_AREA)
                cv2.imwrite(imgPath + str(d) + '.jpg', frame_resize)  # 存储为图像
                print(imgPath + str(d) + '.jpg')
                # cv2.waitKey(5)
        else:
            break
    vc.release()




#video2img('/home/uftp/DJI_0480.MOV','/home/uftp/testimg/')

def jpg2video(soc_dir,dest_dir):
    videoWriter = cv2.VideoWriter(dest_dir, cv2.VideoWriter_fourcc(*'MJPG'), 15,
                                  (720, 480))
    b = os.listdir(soc_dir)
    count = list(filter(lambda x: x[-4:] == '.jpg', b))
    count.sort(key=lambda x: int(x[-8:-4]))
    for i in range(0, len(count)):
        # load pictures from your path
        img = cv2.imread(os.path.join(soc_dir, count[i]))
        img = cv2.resize(img, (720, 480))
        videoWriter.write(img)
        print(os.path.join(soc_dir,count[i]))


    videoWriter.release()

def changeJpgSize(soc_dir,dest_dir):
    os.makedirs(dest_dir, 0o777, True)
    count = 1
    for i in range(1, 20000):
        print(i%5)

        if  i%5 == 0 :
            continue
        else:
            if (os.path.exists(soc_dir + str(i) + '.jpg')):
                # load pictures from your path
                im1 = cv2.imread(soc_dir + str(i) + '.jpg')
                im2 = cv2.resize(im1, (800, 450))  # 为图片重新指定尺寸
                cv2.imwrite(dest_dir+ str(count) + '.jpg', im2)
                print(dest_dir + str(count) + '.jpg')
                count+=1
            else:
                break
    txtname = 'DJI_0003.txt'
    shutil.copyfile(soc_dir + txtname,dest_dir+ txtname)

if __name__ == "__main__":

    '''
    changeJpgSize('/home/wwwroot/default/public/uploads/file/2019-05-13/1557717101/'
    , '/home/wwwroot/default/public/uploads/file/2019-05-13/3/')
    
    #jpg2video('/home/dona/uftp/20190426/0017/','/home/dona/uftp/20190426/result0017.avi')
    # change your path; 30 is fps; (2304,1296) is screen size
    '''
    jpg2video('D:\demo\seg_icnet-master\seg_icnet-master\input\DJI_0177','0117.avi')
    # b = os.listdir('/home/wwwroot/default/public/uploads/file/2019-07-01/1561967018/')
    # count = list(filter(lambda x: x[-8:] == '.jpg.jpg', b))
    # count.sort(key=lambda x: int(x[:-8]))
    # print(count)

