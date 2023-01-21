import cv2
import numpy as np
from matplotlib import pyplot as plt



import cv2
import numpy as np
def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point


    

def rotate_img_and_point1(img_path,points,angle,center_x=None,center_y=None,resize_rate=1.0):



    img = cv2.imread(img_path)
    h,w,c = img.shape

    height, width, channels = img.shape

    center = (width / 2, height / 2)

    # angle = random.uniform(-30, 30)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # rotation_matrix = np.array(rotation_matrix)

    res_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))




    # (cX, cY) = (w / 2, h / 2)
 
    # M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
 
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
 
    # M[0, 2] += (nW / 2) - cX

    # res_img = cv2.warpAffine(img, M, (nW,nH))

    points_in = []

    for point in points:

        points_in.append([point[0][0],point[0][1]])
        points_in.append([point[0][2], point[0][3]])
        points_in.append([point[0][4], point[0][5]])
        points_in.append([point[0][6], point[0][7]])

    out_points = rotate(points_in,rotation_matrix)


    out_points_qbox = []
    qbox = [[]]
    for point in out_points:
        qbox[0].append(point[0])
        qbox[0].append(point[1]) 

        if qbox[0].__len__() == 8:
            out_points_qbox.append(qbox)
            qbox = [[]]


    return res_img,out_points_qbox


def rotate_img_and_point(img_path,points,angle,center_x=None,center_y=None,resize_rate=1.0):

    img = cv2.imread(img_path)

    h,w,c = img.shape
    # center_x = w / 2
    # center_y = h / 2

    # (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    M[0, 2] += (nW / 2) - cX


    # M = cv2.getRotationMatrix2D((center_x,center_y), angle, resize_rate)
    # rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    # rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))


    res_img = cv2.warpAffine(img, M, (nW,nH))

    # res_img = cv2.warpAffine(img, M, (int(w*resize_rate), int(h*resize_rate)))

    points_in = []

    for point in points:

        points_in.append([point[0][0],point[0][1]])
        points_in.append([point[0][2], point[0][3]])
        points_in.append([point[0][4], point[0][5]])
        points_in.append([point[0][6], point[0][7]])

    out_points = rotate(points_in,M)


    out_points_qbox = []
    qbox = [[]]
    for point in out_points:
        qbox[0].append(point[0])
        qbox[0].append(point[1]) 

        if qbox[0].__len__() == 8:
            out_points_qbox.append(qbox)
            qbox = [[]]


    return res_img,out_points_qbox


data_root = 'data/icdar2019_cTDaRA_modern_qbox/'

import mmengine
import mmcv
ann_path = 'data/icdar2019_cTDaRA_modern_qbox/train.json'

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

    with open('data/icdar2019_cTDaRA_modern_qbox/train_qbox/'+file_name.replace('.jpg','.txt'), 'w') as f:
        f.write('\n'.join(lines))


    import random

    # 变化
    # dst, qboxs = rotate_img_and_point(data_root+'train_img/'+file_name, qboxs,30*(random.random()-0.5))
    dst, qboxs = rotate_img_and_point1(data_root+'train_img/'+file_name, qboxs,360*(random.random()))

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
    cv2.imwrite(data_root+'train_rotate_img/'+file_name.replace('.jpg','.png'), dst)

    with open('data/icdar2019_cTDaRA_modern_qbox/train_rotate_qbox/'+file_name.replace('.jpg','.txt'), 'w') as f:
        f.write('\n'.join(lines))



# p1 = plt.subplot(211)
# p1.show(img)
# p1.set_title('Input')

# p2 = plt.subplot(212)
# p2.show(dst)
# p2.set_title('Output')

# plt.show()