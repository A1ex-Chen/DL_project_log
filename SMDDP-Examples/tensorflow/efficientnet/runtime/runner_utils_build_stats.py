def build_stats(history, validation_output, train_callbacks, eval_callback,
    logger):
    stats = {}
    if validation_output:
        stats['eval_loss'] = float(validation_output[0])
        stats['eval_accuracy_top_1'] = float(validation_output[1])
        stats['eval_accuracy_top_5'] = float(validation_output[2])
    if history and history.history:
        train_hist = history.history
        stats['training_loss'] = float(sdp.oob_allreduce(tf.constant(
            train_hist['loss'][-1], dtype=tf.float32)))
        if 'categorical_accuracy' in train_hist:
            stats['training_accuracy_top_1'] = float(sdp.oob_allreduce(tf.
                constant(train_hist['categorical_accuracy'][-1], dtype=tf.
                float32)))
        elif 'sparse_categorical_accuracy' in train_hist:
            stats['training_accuracy_top_1'] = float(sdp.oob_allreduce(tf.
                constant(train_hist['sparse_categorical_accuracy'][-1],
                dtype=tf.float32)))
        elif 'accuracy' in train_hist:
            stats['training_accuracy_top_1'] = float(sdp.oob_allreduce(tf.
                constant(train_hist['accuracy'][-1], dtype=tf.float32)))
            stats['training_accuracy_top_5'] = float(sdp.oob_allreduce(tf.
                constant(train_hist['top_5_accuracy'][-1], dtype=tf.float32)))
    if train_callbacks:
        for callback in train_callbacks:
            if isinstance(callback, callbacks.TimeHistory):
                if callback.epoch_runtime_log:
                    stats['avg_exp_per_second_training'
                        ] = callback.average_examples_per_second
                    stats['avg_exp_per_second_training_per_GPU'
                        ] = callback.average_examples_per_second / sdp.size()
    if eval_callback:
        stats['avg_exp_per_second_eval'] = float(eval_callback.
            average_examples_per_second) * sdp.size()
        stats['avg_exp_per_second_eval_per_GPU'] = float(eval_callback.
            average_examples_per_second)
        stats['avg_time_per_exp_eval'] = 1000.0 / stats[
            'avg_exp_per_second_eval']
        batch_time = eval_callback.batch_time
        batch_time.sort()
        latency_pct_per_batch = sum(batch_time[:-1]) / int(len(batch_time) - 1)
        stats['latency_pct'] = 1000.0 * latency_pct_per_batch
        latency_90pct_per_batch = sum(batch_time[:int(0.9 * len(batch_time))]
            ) / int(0.9 * len(batch_time))
        stats['latency_90pct'] = 1000.0 * latency_90pct_per_batch
        latency_95pct_per_batch = sum(batch_time[:int(0.95 * len(batch_time))]
            ) / int(0.95 * len(batch_time))
        stats['latency_95pct'] = 1000.0 * latency_95pct_per_batch
        latency_99pct_per_batch = sum(batch_time[:int(0.99 * len(batch_time))]
            ) / int(0.99 * len(batch_time))
        stats['latency_99pct'] = 1000.0 * latency_99pct_per_batch
    if not sdp_utils.is_using_sdp() or sdp.rank() == 0:
        logger.log(step=(), data=stats)
