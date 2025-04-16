def eval_trt(self, engine, stride=32):
    self.stride = stride

    def init_engine(engine):
        import tensorrt as trt
        from collections import namedtuple, OrderedDict
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data',
            'ptr'))
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(
                self.device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.
                data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        return context, bindings, binding_addrs, model.get_binding_shape(0)[0]

    def init_data(dataloader, task):
        self.is_coco = self.data.get('is_coco', False)
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(
            range(1000))
        pad = 0.0
        dataloader = create_dataloader(self.data[task if task in ('train',
            'val', 'test') else 'val'], self.img_size, self.batch_size,
            self.stride, check_labels=True, pad=pad, rect=False, data_dict=
            self.data, task=task)[0]
        return dataloader

    def convert_to_coco_format_trt(nums, boxes, scores, classes, paths,
        shapes, ids):
        pred_results = []
        for i, (num, detbox, detscore, detcls) in enumerate(zip(nums, boxes,
            scores, classes)):
            n = int(num[0])
            if n == 0:
                continue
            path, shape = Path(paths[i]), shapes[i][0]
            gain = shapes[i][1][0][0]
            pad = torch.tensor(shapes[i][1][1] * 2).to(self.device)
            detbox = detbox[:n, :]
            detbox -= pad
            detbox /= gain
            detbox[:, 0].clamp_(0, shape[1])
            detbox[:, 1].clamp_(0, shape[0])
            detbox[:, 2].clamp_(0, shape[1])
            detbox[:, 3].clamp_(0, shape[0])
            detbox[:, 2:] = detbox[:, 2:] - detbox[:, :2]
            detscore = detscore[:n]
            detcls = detcls[:n]
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            for ind in range(n):
                category_id = ids[int(detcls[ind])]
                bbox = [round(x, 3) for x in detbox[ind].tolist()]
                score = round(detscore[ind].item(), 5)
                pred_data = {'image_id': image_id, 'category_id':
                    category_id, 'bbox': bbox, 'score': score}
                pred_results.append(pred_data)
        return pred_results
    context, bindings, binding_addrs, trt_batch_size = init_engine(engine)
    assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
    tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self
        .device)
    for _ in range(10):
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
    dataloader = init_data(None, 'val')
    self.speed_result = torch.zeros(4, device=self.device)
    pred_results = []
    pbar = tqdm(dataloader, desc='Inferencing model in validation dataset.',
        ncols=NCOLS)
    for imgs, targets, paths, shapes in pbar:
        nb_img = imgs.shape[0]
        if nb_img != self.batch_size:
            zeros = torch.zeros(self.batch_size - nb_img, 3, *imgs.shape[2:])
            imgs = torch.cat([imgs, zeros], 0)
        t1 = time_sync()
        imgs = imgs.to(self.device, non_blocking=True)
        imgs = imgs.float()
        imgs /= 255
        self.speed_result[1] += time_sync() - t1
        t2 = time_sync()
        binding_addrs['images'] = int(imgs.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        nums = bindings['num_dets'].data[:nb_img]
        boxes = bindings['det_boxes'].data[:nb_img]
        scores = bindings['det_scores'].data[:nb_img]
        classes = bindings['det_classes'].data[:nb_img]
        self.speed_result[2] += time_sync() - t2
        self.speed_result[3] += 0
        pred_results.extend(convert_to_coco_format_trt(nums, boxes, scores,
            classes, paths, shapes, self.ids))
        self.speed_result[0] += self.batch_size
    return dataloader, pred_results
