from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec

from resobj import ResObjRetinaNet, ResObjRetinaNetHead

configs = model_zoo.get_config("COCO-Detection/retinanet_R_50_FPN_1x.py")
train = configs.train
optimizer = configs.optimizer
dataloader = configs.dataloader
lr_multiplier = configs.lr_multiplier

# resobj retinanet
model = L(ResObjRetinaNet)(
    backbone=configs.model.backbone,
    head=L(ResObjRetinaNetHead)(
        # Shape for each input feature map
        input_shape=configs.model.head.input_shape,
        num_classes="${..num_classes}",
        conv_dims=configs.model.head.conv_dims,
        prior_prob=0.001,
        num_anchors=configs.model.head.num_anchors,
    ),
    anchor_generator=configs.model.anchor_generator,
    box2box_transform=configs.model.box2box_transform,
    anchor_matcher=configs.model.anchor_matcher,
    num_classes=80,
    head_in_features=configs.model.head_in_features,
    pixel_mean=configs.model.pixel_mean,
    pixel_std=configs.model.pixel_std,
    input_format="BGR",
)

