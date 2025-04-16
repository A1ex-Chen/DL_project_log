def _load_optimizer_and_scheduler(self, model_path):
    """If optimizer and scheduler states exist, load them."""
    if model_path is not None and os.path.isfile(os.path.join(model_path,
        'optimizer.pt')) and os.path.isfile(os.path.join(model_path,
        'scheduler.pt')):
        if is_torch_tpu_available():
            optimizer_state = torch.load(os.path.join(model_path,
                'optimizer.pt'), map_location='cpu')
            with warnings.catch_warnings(record=True) as caught_warnings:
                lr_scheduler_state = torch.load(os.path.join(model_path,
                    'scheduler.pt'), map_location='cpu')
            reissue_pt_warnings(caught_warnings)
            xm.send_cpu_data_to_device(optimizer_state, self.args.device)
            xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)
            self.optimizer.load_state_dict(optimizer_state)
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        else:
            self.optimizer.load_state_dict(torch.load(os.path.join(
                model_path, 'optimizer.pt'), map_location=self.args.device))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(
                    model_path, 'scheduler.pt')))
            reissue_pt_warnings(caught_warnings)
