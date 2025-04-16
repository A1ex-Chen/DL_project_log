@try_export
def export_torchscript(self, prefix=colorstr('TorchScript:')):
    """YOLOv8 TorchScript model export."""
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...'
        )
    f = self.file.with_suffix('.torchscript')
    ts = torch.jit.trace(self.model, self.im, strict=False)
    extra_files = {'config.txt': json.dumps(self.metadata)}
    if self.args.optimize:
        LOGGER.info(f'{prefix} optimizing for mobile...')
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f),
            _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None
