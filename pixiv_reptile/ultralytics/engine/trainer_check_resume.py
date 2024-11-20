def check_resume(self, overrides):
    """Check if resume checkpoint exists and update arguments accordingly."""
    resume = self.args.resume
    if resume:
        try:
            exists = isinstance(resume, (str, Path)) and Path(resume).exists()
            last = Path(check_file(resume) if exists else get_latest_run())
            ckpt_args = attempt_load_weights(last).args
            if not Path(ckpt_args['data']).exists():
                ckpt_args['data'] = self.args.data
            resume = True
            self.args = get_cfg(ckpt_args)
            self.args.model = self.args.resume = str(last)
            for k in ('imgsz', 'batch', 'device'):
                if k in overrides:
                    setattr(self.args, k, overrides[k])
        except Exception as e:
            raise FileNotFoundError(
                "Resume checkpoint not found. Please pass a valid checkpoint to resume from, i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
    self.resume = resume
