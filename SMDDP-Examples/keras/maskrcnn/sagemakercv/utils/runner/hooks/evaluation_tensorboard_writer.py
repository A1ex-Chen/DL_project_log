def tensorboard_writer(self, stat_dict, iteration, tensorboard_dir):
    writer = tf.summary.create_file_writer(tensorboard_dir)
    with writer.as_default():
        for iou, metric_dict in stat_dict.items():
            for metric_name, metric in metric_dict.items():
                tag = '{}/{}/{}'.format('eval', iou, metric_name)
                tf.summary.scalar(tag, metric, step=iteration)
    writer.close()
