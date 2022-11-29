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

def rotate_img_and_point(img,points,angle,center_x,center_y,resize_rate=1.0):
    h,w,c = img.shape
    M = cv2.getRotationMatrix2D((center_x,center_y), angle, resize_rate)
    res_img = cv2.warpAffine(img, M, (w, h))
    out_points = rotate(points,M)
    return res_img,out_points


data_root = 'data/icdar2019_tracka_modern_qbox/'

import mmengine
import mmcv
ann_path = 'data/icdar2019_tracka_modern_qbox/train.json'

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

    with open('data/icdar2019_tracka_modern_qbox/test_qbox/'+file_name.replace('.jpg','.txt'), 'w') as f:
        f.write('\n'.join(lines))



    # 变化
    dst, qboxs = trans(data_root+'test_img/'+file_name, qboxs)

    lines = []
    for qbox in qboxs:
        type = 'table 0'
        t = ''
        for item in qbox[0]:
            t += str(float(item)) + ' '
        t += type
        print(qbox)
        lines.append(t)

    # cv2.imwrite(data_root+'test_change_img/'+file_name, dst)
    cv2.imwrite(data_root+'test_change_img/'+file_name.replace('.jpg','.png'), dst)

    with open('data/icdar2019_tracka_modern_qbox/test_change_qbox/'+file_name.replace('.jpg','.txt'), 'w') as f:
        f.write('\n'.join(lines))



# p1 = plt.subplot(211)
# p1.show(img)
# p1.set_title('Input')

# p2 = plt.subplot(212)
# p2.show(dst)
# p2.set_title('Output')

# plt.show()