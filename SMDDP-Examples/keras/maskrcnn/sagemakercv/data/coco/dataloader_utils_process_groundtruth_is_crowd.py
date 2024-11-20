def process_groundtruth_is_crowd(data):
    return tf.cond(pred=tf.greater(tf.size(input=data[
        'groundtruth_is_crowd']), 0), true_fn=lambda : data[
        'groundtruth_is_crowd'], false_fn=lambda : tf.zeros_like(data[
        'groundtruth_classes'], dtype=tf.bool))
