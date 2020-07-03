from models.deeplab.backbone import drn, resnet, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, pretrained=False):
    if backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained=pretrained)
    elif backbone == 'resnet': 
        return resnet.ResNet18(output_stride=output_stride, BatchNorm=BatchNorm, pretrained=pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained)
    else:    
        raise NotImplementedError