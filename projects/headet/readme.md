# heading detection

本项目尝试支持从HBBox、RBBox到OBBox对象的检测

# 数据集准备

本仓库在 [ICDAR2019_cTDaRA Modern](https://github.com/cndplab-founder/ICDAR2019_cTDaR)数据集基础上生成而来。


```shell
git clone https://github.com/vansin/ICDAR2019_cTDaR.git -b new ICDAR2019_MTD_HOQ
```

## 数据集config说明

训练集格式为RBBox，测试集格式也为RBBox
project/headet/configs/_base_/datasets/ic19-rbb-rbb.py

训练集格式为OBBox，测试集格式也为OBBox
project/headet/configs/_base_/datasets/ic19-obb-obb.py

训练集格式为HBBox，测试集格式也为OBBox
project/headet/configs/_base_/datasets/ic19-hbb-obb.py

训练集格式为QBBox，测试集格式也为QBBox
project/headet/configs/_base_/datasets/ic19-qbb-qbb.py


## 数据集可视化

```shell
python projects/headet/tools/browse_dataset.py configs/gliding_vertex/gliding-vertex-qbox_r50_fpn_1x_dota.py --stage test
python projects/headet/tools/browse_dataset.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py --stage train
```



# 调试

```shell
python -m debugpy --wait-for-client --listen 5678 tools/train.py projects/headet/configs/rotated_retinanet/rotated-retinanet-hbox-le90_r50_fpn_1x_dota.py
python -m debugpy --wait-for-client --listen 5678 tools/train.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py
```

```shell
 python -m debugpy --wait-for-client --listen 5678 projects/headet/tools/browse_dataset.py projects/headet/configs/gliding_vertex/gliding-vertex-qbox_r50_fpn_1x_dota.py
 python -m debugpy --wait-for-client --listen 5678 projects/headet/tools/browse_dataset.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py --stage train
 ```