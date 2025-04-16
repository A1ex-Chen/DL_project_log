@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr(
    'TorchScript:')):
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...'
        )
    f = file.with_suffix('.torchscript')
    ts = torch.jit.trace(model, im, strict=False)
    d = {'shape': im.shape, 'stride': int(max(model.stride)), 'names':
        model.names}
    extra_files = {'config.txt': json.dumps(d)}
    if optimize:
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f),
            _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None
