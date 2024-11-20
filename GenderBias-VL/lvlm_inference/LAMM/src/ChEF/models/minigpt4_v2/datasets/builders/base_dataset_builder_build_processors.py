def build_processors(self):
    vis_proc_cfg = self.config.get('vis_processor')
    txt_proc_cfg = self.config.get('text_processor')
    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get('train')
        vis_eval_cfg = vis_proc_cfg.get('eval')
        self.vis_processors['train'] = self._build_proc_from_cfg(vis_train_cfg)
        self.vis_processors['eval'] = self._build_proc_from_cfg(vis_eval_cfg)
    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get('train')
        txt_eval_cfg = txt_proc_cfg.get('eval')
        self.text_processors['train'] = self._build_proc_from_cfg(txt_train_cfg
            )
        self.text_processors['eval'] = self._build_proc_from_cfg(txt_eval_cfg)
