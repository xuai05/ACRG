# ACRG: Agent-Centric Relation Graph for Object Visual Navigation


### 调试代码
python main.py --gpu-ids 0  --workers 1 --model ACRGModel --detr --title debug_code --work-dir ./work_dirs/ --test-after-train --pretrained-trans /data/huxiaobo/Data/pretrain_dirs/pretrain_vistrans_train_2021-06-09_10-47-35/trained_models/checkpoint0004.pth --ep-save-freq 10


### 测试代码是否可运行
python main.py --gpu-ids 0 1 2  --workers 9 --model ACRGModel --detr --title test --work-dir ./work_dirs/ --test-after-train --pretrained-trans /data/huxiaobo/Data/pretrain_dirs/pretrain_vistrans_train_2021-06-09_10-47-35/trained_models/checkpoint0004.pth --ep-save-freq 100000




### 测试不同垂直信息添加方式——不构建多余的图，而是直接将垂直信息添加到水平关系图上，构建一个水平垂直关系图和深度关系图
python main.py --gpu-ids 0 1 2  --workers 15 --model ACRGModel --detr --title ACRG_depth_graph_and_vertical_in_horization_graph --work-dir ./work_dirs/ --test-after-train --pretrained-trans /data/huxiaobo/Data/pretrain_dirs/pretrain_vistrans_train_2021-06-09_10-47-35/trained_models/checkpoint0004.pth --ep-save-freq 100000