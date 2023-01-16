# heading detection

本项目尝试支持从HBBox、RBBox到OBBox对象的检测

# browse_dataset prepare

本仓库在 [ICDAR2019_cTDaRA Modern](https://github.com/cndplab-founder/ICDAR2019_cTDaR)数据集基础上生成而来。


## 数据集可视化

```shell
python projects/headet/tools/browse_dataset.py configs/gliding_vertex/gliding-vertex-qbox_r50_fpn_1x_dota.py --stage test
```



# 调试

```shell
python -m debugpy --wait-for-client --listen 5678 tools/train.py projects/headet/configs/rotated_retinanet/rotated-retinanet-hbox-le90_r50_fpn_1x_dota.py
python -m debugpy --wait-for-client --listen 5678 tools/train.py projects/headet/configs/rotated_retinanet/rotated-retinanet-rbox-h180_r50_fpn_1x_dota.py
```

```shell
 python -m debugpy --wait-for-client --listen 5678 projects/headet/tools/browse_dataset.py projects/headet/configs/gliding_vertex/gliding-vertex-qbox_r50_fpn_1x_dota.py
 ```