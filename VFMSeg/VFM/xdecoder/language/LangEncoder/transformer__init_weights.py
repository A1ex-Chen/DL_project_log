def _init_weights(self, m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if is_main_process():
            logger.info('=> init weight of Linear/Conv2d from trunc norm')
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            if is_main_process():
                logger.info('=> init bias of Linear/Conv2d to zeros')
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.bias, 0)
