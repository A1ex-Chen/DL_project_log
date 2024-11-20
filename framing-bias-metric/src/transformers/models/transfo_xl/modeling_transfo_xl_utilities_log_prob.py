def log_prob(self, hidden):
    """
        Computes log probabilities for all :math:`n\\_classes` From:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.p

        Args:
            hidden (Tensor): a minibatch of example

        Returns:
            log-probabilities of for each class :math:`c` in range :math:`0 <= c <= n\\_classes`, where
            :math:`n\\_classes` is a parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor. Shape:

            - Input: :math:`(N, in\\_features)`
            - Output: :math:`(N, n\\_classes)`
        """
    if self.n_clusters == 0:
        logit = self._compute_logit(hidden, self.out_layers[0].weight, self
            .out_layers[0].bias, self.out_projs[0])
        return F.log_softmax(logit, dim=-1)
    else:
        weights, biases = [], []
        for i in range(len(self.cutoffs)):
            if self.div_val == 1:
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                weight_i = self.out_layers[0].weight[l_idx:r_idx]
                bias_i = self.out_layers[0].bias[l_idx:r_idx]
            else:
                weight_i = self.out_layers[i].weight
                bias_i = self.out_layers[i].bias
            if i == 0:
                weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
            weights.append(weight_i)
            biases.append(bias_i)
        head_weight, head_bias, head_proj = weights[0], biases[0
            ], self.out_projs[0]
        head_logit = self._compute_logit(hidden, head_weight, head_bias,
            head_proj)
        out = hidden.new_empty((head_logit.size(0), self.n_token))
        head_logprob = F.log_softmax(head_logit, dim=1)
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]
            if i == 0:
                out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]]
            else:
                weight_i, bias_i, proj_i = weights[i], biases[i
                    ], self.out_projs[i]
                tail_logit_i = self._compute_logit(hidden, weight_i, bias_i,
                    proj_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                logprob_i = head_logprob[:, -i] + tail_logprob_i
                out[:, start_idx, stop_idx] = logprob_i
        return out
