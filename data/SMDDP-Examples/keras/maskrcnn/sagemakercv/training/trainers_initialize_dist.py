def initialize_dist(self, dist):
    if dist is None:
        return
    if dist.lower() in ['hvd', 'horovod']:
        logging.info('Using Horovod For Distributed Training')
        import horovod.tensorflow as dist
        dist.init()
        return dist
    elif dist.lower() in ['smd', 'sagemaker', 'smddp']:
        logging.info('Using Sagemaker For Distributed Training')
        import smdistributed.dataparallel.tensorflow as dist
        dist.init()
        return dist
    else:
        raise NotImplementedError
