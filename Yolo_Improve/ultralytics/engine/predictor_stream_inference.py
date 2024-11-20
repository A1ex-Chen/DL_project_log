@smart_inference_mode()
def stream_inference(self, source=None, model=None, *args, **kwargs):
    """Streams real-time inference on camera feed and saves results to file."""
    if self.args.verbose:
        LOGGER.info('')
    if not self.model:
        self.setup_model(model)
    with self._lock:
        self.setup_source(source if source is not None else self.args.source)
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir
                ).mkdir(parents=True, exist_ok=True)
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.
                triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True
        self.seen, self.windows, self.batch = 0, [], None
        profilers = ops.Profile(device=self.device), ops.Profile(device=
            self.device), ops.Profile(device=self.device)
        self.run_callbacks('on_predict_start')
        for self.batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            paths, im0s, s = self.batch
            with profilers[0]:
                im = self.preprocess(im0s)
            with profilers[1]:
                preds = self.inference(im, *args, **kwargs)
                if self.args.embed:
                    yield from ([preds] if isinstance(preds, torch.Tensor) else
                        preds)
                    continue
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')
            n = len(im0s)
            for i in range(n):
                self.seen += 1
                self.results[i].speed = {'preprocess': profilers[0].dt * 
                    1000.0 / n, 'inference': profilers[1].dt * 1000.0 / n,
                    'postprocess': profilers[2].dt * 1000.0 / n}
                if (self.args.verbose or self.args.save or self.args.
                    save_txt or self.args.show):
                    s[i] += self.write_results(i, Path(paths[i]), im, s)
            if self.args.verbose:
                LOGGER.info('\n'.join(s))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results
    for v in self.vid_writer.values():
        if isinstance(v, cv2.VideoWriter):
            v.release()
    if self.args.verbose and self.seen:
        t = tuple(x.t / self.seen * 1000.0 for x in profilers)
        LOGGER.info(
            f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape {min(self.args.batch, self.seen), 3, *im.shape[2:]}'
             % t)
    if self.args.save or self.args.save_txt or self.args.save_crop:
        nl = len(list(self.save_dir.glob('labels/*.txt')))
        s = (
            f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}"
             if self.args.save_txt else '')
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
    self.run_callbacks('on_predict_end')
