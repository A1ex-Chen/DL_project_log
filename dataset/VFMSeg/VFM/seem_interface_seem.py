def interface_seem(model, audio, image, tasks, info, refimg=None, reftxt=
    None, audio_pth=None, video_pth=None):
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    all_classes = [name.replace('-other', '').replace('-merged', '') for
        name in COCO_PANOPTIC_CLASSES] + ['others']
    colors_list = [(np.array(color['color']) / 255).tolist() for color in
        COCO_CATEGORIES] + [[1, 1, 1]]
    image_ori = image
    mapping = info[0]
    prompt = info[1]
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    data = {'image': images, 'height': height, 'width': width}
    if len(tasks) == 0:
        tasks = ['Panoptic']
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False
    example = None
    stroke = None
    text = None
    audio = None
    batch_inputs = [data]
    if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results = model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        pano_seg_logits = results[-1]['panoptic_seg'][2].permute(2, 0, 1)
        if mapping == 'NuScenesLidarSegSCN':
            pano_seg_logits = pano_seg_logits[COCO_TO_NuScenes['masks']]
            mapped_labels = np.array(COCO_TO_NuScenes['classes'])[
                COCO_TO_NuScenes['masks']]
            class_num = COCO_TO_NuScenes['class_num']
            CoCoMap = COCO_TO_NuScenes['Mapping']
        elif mapping == 'A2D2SCN':
            pano_seg_logits = pano_seg_logits[COCO_TO_A2D2_SKITTI['masks']]
            mapped_labels = np.array(COCO_TO_A2D2_SKITTI['classes'])[
                COCO_TO_A2D2_SKITTI['masks']]
            class_num = COCO_TO_A2D2_SKITTI['class_num']
            CoCoMap = COCO_TO_A2D2_SKITTI['Mapping']
        elif mapping == 'SemanticKITTISCN':
            pano_seg_logits = pano_seg_logits[COCO_TO_VKITTI_SKITTI['masks']]
            mapped_labels = np.array(COCO_TO_VKITTI_SKITTI['classes'])[
                COCO_TO_VKITTI_SKITTI['masks']]
            class_num = COCO_TO_VKITTI_SKITTI['class_num']
            CoCoMap = COCO_TO_VKITTI_SKITTI['Mapping']
        else:
            pass
        res = torch.zeros((class_num, pano_seg_logits.shape[1],
            pano_seg_logits.shape[2]), dtype=torch.half, device=
            pano_seg_logits.device) - 100.0
        for i in range(class_num):
            mask = mapped_labels == i
            check = set(mask)
            if 1 == len(check) and False in check:
                continue
            res[i] = torch.mean(pano_seg_logits[mask], axis=0)
        for mask in pano_seg_info:
            k = mask['category_id'] + 1
            if k in CoCoMap.keys():
                pano_seg[pano_seg == mask['id']] = CoCoMap[k]
            else:
                pano_seg[pano_seg == mask['id']] = -100
        return res, pano_seg
    else:
        results, image_size, extra = model.model.evaluate_demo(batch_inputs)
    res = []
    return Image.fromarray(res), None
