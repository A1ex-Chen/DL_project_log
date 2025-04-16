@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr('PaddlePaddle:')):
    check_requirements(('paddlepaddle', 'x2paddle'))
    import x2paddle
    from x2paddle.convert import pytorch2paddle
    LOGGER.info(
        f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
    f = str(file).replace('.pt', f'_paddle_model{os.sep}')
    pytorch2paddle(module=model, save_dir=f, jit_type='trace',
        input_examples=[im])
    return f, None
