def call(self, hidden, target, return_mean=True, training=False):
    head_logprob = 0
    if self.n_clusters == 0:
        output = self._logit(hidden, self.out_layers[0][0], self.out_layers
            [0][1], self.out_projs[0])
        if target is not None:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
                target, logits=output)
        out = tf.nn.log_softmax(output, axis=-1)
    else:
        hidden_sizes = shape_list(hidden)
        out = []
        loss = tf.zeros(hidden_sizes[:2], dtype=tf.float32)
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            if target is not None:
                mask = (target >= l_idx) & (target < r_idx)
                mask_idx = tf.where(mask)
                cur_target = tf.boolean_mask(target, mask) - l_idx
            if self.div_val == 1:
                cur_W = self.out_layers[0][0][l_idx:r_idx]
                cur_b = self.out_layers[0][1][l_idx:r_idx]
            else:
                cur_W = self.out_layers[i][0]
                cur_b = self.out_layers[i][1]
            if i == 0:
                cur_W = tf.concat([cur_W, self.cluster_weight], 0)
                cur_b = tf.concat([cur_b, self.cluster_bias], 0)
                head_logit = self._logit(hidden, cur_W, cur_b, self.
                    out_projs[0])
                head_logprob = tf.nn.log_softmax(head_logit)
                out.append(head_logprob[..., :self.cutoffs[0]])
                if target is not None:
                    cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                    cur_logprob = self._gather_logprob(cur_head_logprob,
                        cur_target)
            else:
                tail_logit = self._logit(hidden, cur_W, cur_b, self.
                    out_projs[i])
                tail_logprob = tf.nn.log_softmax(tail_logit)
                cluster_prob_idx = self.cutoffs[0] + i - 1
                logprob_i = head_logprob[..., cluster_prob_idx, None
                    ] + tail_logprob
                out.append(logprob_i)
                if target is not None:
                    cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                    cur_tail_logprob = tf.boolean_mask(tail_logprob, mask)
                    cur_logprob = self._gather_logprob(cur_tail_logprob,
                        cur_target)
                    cur_logprob += cur_head_logprob[:, self.cutoff_ends[1] +
                        i - 1]
            if target is not None:
                loss += tf.scatter_nd(mask_idx, -cur_logprob, tf.cast(
                    shape_list(loss), dtype=tf.int64))
        out = tf.concat(out, axis=-1)
    if target is not None:
        if return_mean:
            loss = tf.reduce_mean(loss)
        self.add_loss(loss)
        self.add_metric(loss, name=self.name, aggregation='mean' if
            return_mean else '')
    return out
