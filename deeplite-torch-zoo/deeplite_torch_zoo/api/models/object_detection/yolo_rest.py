from deeplite_torch_zoo.src.object_detection.yolo import YOLO
from deeplite_torch_zoo.api.models.object_detection.helpers import (
    make_wrapper_func, load_pretrained_model, get_project_root
)
from deeplite_torch_zoo.api.datasets.object_detection.yolo import DATASET_CONFIGS

__all__ = []

CFG_PATH = 'deeplite_torch_zoo/src/object_detection/yolo/configs'

YOLO_CONFIGS = {
    'yolo3': 'yolo3/yolov3.yaml',
    'yolo4': 'yolo4/yolov4.yaml',
    'yolo5': 'yolo5/yolov5.yaml',
    'yolo7': 'yolo7/yolov7.yaml',
    'yolo8': 'yolo8/yolov8.yaml',
    ############################
    'yolo3-spp-': 'yolo3/yolov3-spp.yaml',
    'yolo3-tiny-': 'yolo3/yolov3-tiny.yaml',
    ############################
    'yolo4-tiny-': 'yolo4/yolov4-tiny.yaml',
    'yolo4-pacsp-': 'yolo4/yolov4-pacsp.yaml',
    'yolo4-csp-p5': 'yolo4/yolov4-csp-p5.yaml',
    'yolo4-csp-p6': 'yolo4/yolov4-csp-p6.yaml',
    'yolo4-csp-p7': 'yolo4/yolov4-csp-p7.yaml',
    ############################
    'yolo5.6': 'yolo5/yolov5.6.yaml',
    'yolo5-p2': 'yolo5/yolov5-p2.yaml',
    'yolo5-p34': 'yolo5/yolov5-p34.yaml',
    'yolo5-p6': 'yolo5/yolov5-p6.yaml',
    'yolo5-p7': 'yolo5/yolov5-p7.yaml',
    'yolo5-fpn-': 'yolo5/yolov5-fpn.yaml',
    'yolo5-bifpn-': 'yolo5/yolov5-bifpn.yaml',
    'yolo5-ghost-': 'yolo5/yolov5-ghost.yaml',
    'yolo5-panet-': 'yolo5/yolov5-panet.yaml',
    ############################
    'yolor': 'yolor/yolor-csp.yaml',
    'yolor-d6': 'yolor/yolor-d6.yaml',
    'yolor-e6': 'yolor/yolor-e6.yaml',
    'yolor-p6': 'yolor/yolor-p6.yaml',
    'yolor-w6': 'yolor/yolor-w6.yaml',
    'yolor-dwt-': 'yolor/yolor-dwt.yaml',
    'yolor-s2d-': 'yolor/yolor-s2d.yaml',
    ############################
    'yolo7-tiny-': 'yolo7/yolov7-tiny.yaml',
    'yolo7-e6': 'yolo7/yolov7-e6.yaml',
    'yolo7-e6e': 'yolo7/yolov7-e6e.yaml',
    'yolo7-w6': 'yolo7/yolov7-w6.yaml',
    ############################
    'yolo8-p2': 'yolo8/yolov8-p2.yaml',
    'yolo8-p6': 'yolo8/yolov8-p6.yaml',
    ############################
    'yolox': 'yolox/yolox.yaml',
    ############################
    'yolo-r50-csp-': 'misc/r50-csp.yaml',
    # 'yolo-x50-csp-': 'misc/x50-csp.yaml',  # to be fixed
    ############################
    'yolo-picodet-': 'picodet/yolo-picodet.yaml',
    'yolo5-lite-c-': 'yololite/yolov5_lite_c.yaml',
    'yolo5-lite-e-': 'yololite/yolov5_lite_e.yaml',
    ############################
    'edgeyolo-': 'edgeyolo/edgeyolo.yaml',
    'edgeyolo-m-': 'edgeyolo/edgeyolo_m.yaml',
    'edgeyolo-s-': 'edgeyolo/edgeyolo_s.yaml',
    'edgeyolo-tiny-': 'edgeyolo/edgeyolo_tiny.yaml',
    'edgeyolo-tiny-lrelu-': 'edgeyolo/edgeyolo_tiny_lrelu.yaml',
    ############################
    'yolo6u': 'yolo6u/yolov6.yaml',
}

ACT_FN_TAGS = {'': None,'_relu': 'relu','_hswish': 'hardswish'}

YOLO_MCU_CONFIGS = {
    'yolo3': 'yolo3/yolov3.yaml',
    'yolo4': 'yolo4/yolov4.yaml',
    'yolo5': 'yolo5/yolov5.yaml',
    'yolo7': 'yolo7/yolov7.yaml',
    'yolo8': 'yolo8/yolov8.yaml',
    'yolo6u': 'yolo6u/yolov6.yaml',

}

DEFAULT_MODEL_SCALES = {
    # [depth, width, max_channels]
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 1024],
    'l': [1.00, 1.00, 1024],
    'x': [1.00, 1.25, 1024],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 1024],
    'd1w25': [1.0, 0.25, 1024],
    'd1w75': [1.0, 0.75, 1024],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 1024],
    'd67w5': [0.67, 0.5, 1024],
    'd67w25': [0.67, 0.25, 1024],
}

MODEL_SCALES_v8 = {
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 768],
    'l': [1.00, 1.00, 512],
    'x': [1.00, 1.25, 512],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 512],
    'd1w25': [1.0, 0.25, 512],
    'd1w75': [1.0, 0.75, 512],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 768],
    'd67w5': [0.67, 0.5, 768],
    'd67w25': [0.67, 0.25, 768],
    'd250w250' : [0.25, 0.25, 1024],
    'd250w200' : [0.25, 0.2, 1024],
    'd250w160' : [0.25, 0.16, 1024],
    'd250w125' : [0.25, 0.125, 1024],
    'd250w85' : [0.25, 0.085, 1024],
    'd250w50' : [0.25, 0.05, 1024],
    'd200w250' : [0.2, 0.25, 1024],
    'd200w200' : [0.2, 0.2, 1024],
    'd200w160' : [0.2, 0.16, 1024],
    'd200w125' : [0.2, 0.125, 1024],
    'd200w85' : [0.2, 0.085, 1024],
    'd200w50' : [0.2, 0.05, 1024],
}

MODEL_SCALES_v7 = {
        # [depth, width, max_channels]
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 1024],
    'l': [1.00, 1.00, 1024],
    'x': [1.00, 1.25, 1024],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 1024],
    'd1w25': [1.0, 0.25, 1024],
    'd1w75': [1.0, 0.75, 1024],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 1024],
    'd67w5': [0.67, 0.5, 1024],
    'd67w25': [0.67, 0.25, 1024],
    'd250w250' : [0.25, 0.25, 1024],
    'd250w200' : [0.25, 0.2, 1024],
    'd250w160' : [0.25, 0.16, 1024],
    'd250w125' : [0.25, 0.125, 1024],
    'd250w85' : [0.25, 0.085, 1024],
    'd250w50' : [0.25, 0.05, 1024],
}

MODEL_SCALES_v6 = {
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 768],
    'l': [1.00, 1.00, 512],
    'x': [1.00, 1.25, 512],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 512],
    'd1w25': [1.0, 0.25, 512],
    'd1w75': [1.0, 0.75, 512],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 768],
    'd67w5': [0.67, 0.5, 768],
    'd67w25': [0.67, 0.25, 768],
    'd250w250' : [0.25, 0.25, 1024],
    'd250w200' : [0.25, 0.2, 1024],
    'd250w160' : [0.25, 0.16, 1024],
    'd250w125' : [0.25, 0.125, 1024],
    'd250w85' : [0.25, 0.085, 1024],
    'd250w50' : [0.25, 0.05, 1024],
    'd200w250' : [0.2, 0.25, 1024],
    'd200w200' : [0.2, 0.2, 1024],
    'd200w160' : [0.2, 0.16, 1024],
    'd200w125' : [0.2, 0.125, 1024],
    'd200w85' : [0.2, 0.085, 1024],
    'd200w50' : [0.2, 0.05, 1024],
    'd160w250' : [0.16, 0.25, 1024],
    'd160w200' : [0.16, 0.2, 1024],
    'd160w160' : [0.16, 0.16, 1024],
    'd160w125' : [0.16, 0.125, 1024],
    'd160w85' : [0.16, 0.085, 1024],
    'd160w50' : [0.16, 0.05, 1024],
    'd125w250' : [0.125, 0.25, 1024],
    'd125w200' : [0.125, 0.2, 1024],
    'd125w160' : [0.125, 0.16, 1024],
    'd125w125' : [0.125, 0.125, 1024],
    'd125w85' : [0.125, 0.085, 1024],
    'd125w50' : [0.125, 0.05, 1024],
    'd85w250' : [0.085, 0.25, 1024],
    'd85w200' : [0.085, 0.2, 1024],
    'd85w160' : [0.085, 0.16, 1024],
    'd85w125' : [0.085, 0.125, 1024],
    'd85w85' : [0.085, 0.085, 1024],
    'd85w50' : [0.085, 0.05, 1024],
}

MODEL_SCALES_v5 = {
    # [depth, width, max_channels]
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 1024],
    'l': [1.00, 1.00, 1024],
    'x': [1.00, 1.25, 1024],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 1024],
    'd1w25': [1.0, 0.25, 1024],
    'd1w75': [1.0, 0.75, 1024],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 1024],
    'd67w5': [0.67, 0.5, 1024],
    'd67w25': [0.67, 0.25, 1024],
    'd250w250' : [0.25, 0.25, 1024],
    'd250w200' : [0.25, 0.2, 1024],
    'd250w160' : [0.25, 0.16, 1024],
    'd250w125' : [0.25, 0.125, 1024],
    'd250w85' : [0.25, 0.085, 1024],
    'd250w50' : [0.25, 0.05, 1024],
    'd200w250' : [0.2, 0.25, 1024],
    'd200w200' : [0.2, 0.2, 1024],
    'd200w160' : [0.2, 0.16, 1024],
    'd200w125' : [0.2, 0.125, 1024],
    'd200w85' : [0.2, 0.085, 1024],
    'd200w50' : [0.2, 0.05, 1024],
    'd160w250' : [0.16, 0.25, 1024],
    'd160w200' : [0.16, 0.2, 1024],
    'd160w160' : [0.16, 0.16, 1024],
    'd160w125' : [0.16, 0.125, 1024],
    'd160w85' : [0.16, 0.085, 1024],
    'd160w50' : [0.16, 0.05, 1024],
    'd125w250' : [0.125, 0.25, 1024],
    'd125w200' : [0.125, 0.2, 1024],
    'd125w160' : [0.125, 0.16, 1024],
    'd125w125' : [0.125, 0.125, 1024],
    'd125w85' : [0.125, 0.085, 1024],
    'd125w50' : [0.125, 0.05, 1024],
}

CUSTOM_MODEL_SCALES = {'yolo8': MODEL_SCALES_v8, 'yolo7': MODEL_SCALES_v7, 'yolo6u': MODEL_SCALES_v6, 'yolo5': MODEL_SCALES_v5, 'yolo4': MODEL_SCALES_v5, 'yolo3': MODEL_SCALES_v5}






full_model_dict = {}
for model_key, config_name in YOLO_CONFIGS.items():
    for cfg_name, param_dict in get_model_scales(model_key).items():
        for activation_fn_tag, act_fn_name in ACT_FN_TAGS.items():
            full_model_dict[f'{model_key}{cfg_name}{activation_fn_tag}'] = {
                'params': {**param_dict, 'activation_type': act_fn_name},
                'config': get_project_root() / CFG_PATH / config_name,
            }

for dataset_tag, dataset_config in DATASET_CONFIGS.items():
    for model_tag, model_dict in full_model_dict.items():
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(
            create_yolo_model,
            name,
            model_tag,
            dataset_tag,
            dataset_config.num_classes,
            config_path=model_dict['config'],
            **model_dict['params'],
        )
        __all__.append(name)