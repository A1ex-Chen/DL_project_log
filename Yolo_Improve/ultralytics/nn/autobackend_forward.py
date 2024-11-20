def forward(self, im, augment=False, visualize=False, embed=None):
    """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
    b, ch, h, w = im.shape
    if self.fp16 and im.dtype != torch.float16:
        im = im.half()
    if self.nhwc:
        im = im.permute(0, 2, 3, 1)
    if self.pt or self.nn_module:
        y = self.model(im, augment=augment, visualize=visualize, embed=embed)
    elif self.jit:
        y = self.model(im)
    elif self.dnn:
        im = im.cpu().numpy()
        self.net.setInput(im)
        y = self.net.forward()
    elif self.onnx:
        im = im.cpu().numpy()
        y = self.session.run(self.output_names, {self.session.get_inputs()[
            0].name: im})
    elif self.xml:
        im = im.cpu().numpy()
        if self.inference_mode in {'THROUGHPUT', 'CUMULATIVE_THROUGHPUT'}:
            n = im.shape[0]
            results = [None] * n

            def callback(request, userdata):
                """Places result in preallocated list using userdata index."""
                results[userdata] = request.results
            async_queue = self.ov.runtime.AsyncInferQueue(self.
                ov_compiled_model)
            async_queue.set_callback(callback)
            for i in range(n):
                async_queue.start_async(inputs={self.input_name: im[i:i + 1
                    ]}, userdata=i)
            async_queue.wait_all()
            y = np.concatenate([list(r.values())[0] for r in results])
        else:
            y = list(self.ov_compiled_model(im).values())
    elif self.engine:
        if self.dynamic or im.shape != self.bindings['images'].shape:
            if self.is_trt10:
                self.context.set_input_shape('images', im.shape)
                self.bindings['images'] = self.bindings['images']._replace(
                    shape=im.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.
                        get_tensor_shape(name)))
            else:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)
                self.bindings['images'] = self.bindings['images']._replace(
                    shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.
                        get_binding_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
    elif self.coreml:
        im = im[0].cpu().numpy()
        im_pil = Image.fromarray((im * 255).astype('uint8'))
        y = self.model.predict({'image': im_pil})
        if 'confidence' in y:
            raise TypeError(
                f"Ultralytics only supports inference of non-pipelined CoreML models exported with 'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                )
        elif len(y) == 1:
            y = list(y.values())
        elif len(y) == 2:
            y = list(reversed(y.values()))
    elif self.paddle:
        im = im.cpu().numpy().astype(np.float32)
        self.input_handle.copy_from_cpu(im)
        self.predictor.run()
        y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in
            self.output_names]
    elif self.ncnn:
        mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.
                output_names())]
    elif self.triton:
        im = im.cpu().numpy()
        y = self.model(im)
    else:
        im = im.cpu().numpy()
        if self.saved_model:
            y = self.model(im, training=False) if self.keras else self.model(im
                )
            if not isinstance(y, list):
                y = [y]
        elif self.pb:
            y = self.frozen_func(x=self.tf.constant(im))
            if (self.task == 'segment' or len(y) == 2) and len(self.names
                ) == 999:
                ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)
                nc = y[ib].shape[1] - y[ip].shape[3] - 4
                self.names = {i: f'class{i}' for i in range(nc)}
        else:
            details = self.input_details[0]
            is_int = details['dtype'] in {np.int8, np.int16}
            if is_int:
                scale, zero_point = details['quantization']
                im = (im / scale + zero_point).astype(details['dtype'])
            self.interpreter.set_tensor(details['index'], im)
            self.interpreter.invoke()
            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output['index'])
                if is_int:
                    scale, zero_point = output['quantization']
                    x = (x.astype(np.float32) - zero_point) * scale
                if x.ndim == 3:
                    x[:, [0, 2]] *= w
                    x[:, [1, 3]] *= h
                y.append(x)
        if len(y) == 2:
            if len(y[1].shape) != 4:
                y = list(reversed(y))
            y[1] = np.transpose(y[1], (0, 3, 1, 2))
        y = [(x if isinstance(x, np.ndarray) else x.numpy()) for x in y]
    if isinstance(y, (list, tuple)):
        return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x
            ) for x in y]
    else:
        return self.from_numpy(y)
