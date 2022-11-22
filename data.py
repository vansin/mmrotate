import cv2
import numpy as np
from matplotlib import pyplot as plt



def cvt_pos(pos, cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0]*u+cvt_mat_t[0][1]*v+cvt_mat_t[0][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0]*u+cvt_mat_t[1][1]*v+cvt_mat_t[1][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])
    return (int(x), int(y))



def trans(image_path, qboxs):

    img = cv2.imread(image_path)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])


    import random
    x1,y1=0.1*rows*random.random(), 0.1*cols*random.random()
    x2,y2=0.1*rows*random.random(), 0.1*cols*random.random()
    x3,y3=0.1*rows*random.random(), 0.1*cols*random.random()
    x4,y4=0.1*rows*random.random(), 0.1*cols*random.random()
    # x2,y2=10,10
    # x3,y3=10,10
    # x4,y4=10,100


    pts2 = np.float32([[0+x1, 0+x1], [cols-x2, 0+y2], [cols-x3, rows-y3], [0+x4, rows-y4]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))


    for qbox in qboxs:
        x1,y1,x2,y2,x3,y3,x4,y4 = qbox[0]

        x1,y1 = cvt_pos([x1,y1], M)
        x2,y2 = cvt_pos([x2,y2], M)
        x3,y3 = cvt_pos([x3,y3], M)
        x4,y4 = cvt_pos([x4,y4], M)

        qbox[0] = [x1,y1, x2, y2, x3, y3, x4, y4]


    return dst, qboxs

    # data = cvt_pos([368,52], M)

    # print(data)

    # plt.figure(figsize=(8, 7), dpi=98)

    # cv2.imshow('show', dst)

    # cv2.waitKey(0)

data_root = 'data/icdar2019_tracka_modern/'

import mmengine
import mmcv
ann_path = 'data/icdar2019_tracka_modern/train.json'

ann = mmengine.load(ann_path)



categories = ann['categories']
annotations = ann['annotations']
images =  ann['images']


for annotation in annotations:

    image_id = annotation['image_id']
    segmentation = annotation['segmentation']

    if 'qboxs' not in images[image_id]:
        images[image_id]['qboxs'] = [segmentation]
    else:
        images[image_id]['qboxs'].append(segmentation)

    pass




for image in images:

    qboxs = image['qboxs']
    file_name = image['file_name']

    lines = []
    for qbox in qboxs:
        type = 'table 0'
        t = ''
        for item in qbox[0]:
            t += str(float(item)) + ' '
        t += type
        print(qbox)
        lines.append(t)

    with open('data/icdar2019_tracka_modern/train_qbox/'+file_name.replace('.jpg','.txt'), 'w') as f:
        f.write('\n'.join(lines))



    # 变化
    dst, qboxs = trans(data_root+'train_img/'+file_name, qboxs)

    lines = []
    for qbox in qboxs:
        type = 'table 0'
        t = ''
        for item in qbox[0]:
            t += str(float(item)) + ' '
        t += type
        print(qbox)
        lines.append(t)

    # cv2.imwrite(data_root+'train_change_img/'+file_name, dst)
    cv2.imwrite(data_root+'train_change_img/'+file_name.replace('.jpg','.png'), dst)

    with open('data/icdar2019_tracka_modern/train_change_qbox/'+file_name.replace('.jpg','.txt'), 'w') as f:
        f.write('\n'.join(lines))



# p1 = plt.subplot(211)
# p1.show(img)
# p1.set_title('Input')

# p2 = plt.subplot(212)
# p2.show(dst)
# p2.set_title('Output')

# plt.show()