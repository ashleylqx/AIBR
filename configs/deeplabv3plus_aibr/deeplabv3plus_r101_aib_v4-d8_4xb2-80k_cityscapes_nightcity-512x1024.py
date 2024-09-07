_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_aib_v4.py',
    '../_base_/custom_datasets/cityscapes_nightcity.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
            pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
