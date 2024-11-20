@try_export
def export_paddle(self, prefix=colorstr('PaddlePaddle:')):
    """YOLOv8 Paddle export."""
    check_requirements(('paddlepaddle', 'x2paddle'))
    import x2paddle
    from x2paddle.convert import pytorch2paddle
    LOGGER.info(
        f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
    f = str(self.file).replace(self.file.suffix, f'_paddle_model{os.sep}')
    pytorch2paddle(module=self.model, save_dir=f, jit_type='trace',
        input_examples=[self.im])
    yaml_save(Path(f) / 'metadata.yaml', self.metadata)
    return f, None
