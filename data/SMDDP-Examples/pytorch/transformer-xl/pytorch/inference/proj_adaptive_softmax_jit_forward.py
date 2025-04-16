def forward(self, hidden, target, keep_order: bool=False):
    """
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        """
    if hidden.size(0) != target.size(0):
        raise RuntimeError(
            'Input and target should have the same size in the batch dimension.'
            )
    if self.n_clusters == 0:
        logit = self._compute_logit(hidden, self.out_layers_weights[0],
            self.out_layers_biases[0], self.out_projs[0])
        nll = -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)
            ).squeeze(1)
    else:
        weights, biases = [], []
        for i in range(len(self.cutoffs)):
            if self.div_val == 1:
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                weight_i = self.out_layers_weights[0][l_idx:r_idx]
                bias_i = self.out_layers_biases[0][l_idx:r_idx]
            else:
                weight_i = self.out_layers_weights[i]
                bias_i = self.out_layers_biases[i]
            if i == 0:
                weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)
            weights.append(weight_i)
            biases.append(bias_i)
        head_weight, head_bias, head_proj = weights[0], biases[0
            ], self.out_projs[0]
        head_logit = self._compute_logit(hidden, head_weight, head_bias,
            head_proj)
        head_logprob = F.log_softmax(head_logit, dim=1)
        nll = torch.zeros_like(target, layout=torch.strided, dtype=hidden.
            dtype, device=hidden.device)
        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
            mask_i = (target >= l_idx) & (target < r_idx)
            indices_i = mask_i.nonzero().squeeze()
            if indices_i.numel() == 0:
                continue
            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)
            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]
                    ).squeeze(1)
            else:
                weight_i, bias_i, proj_i = weights[i], biases[i
                    ], self.out_projs[i]
                hidden_i = hidden.index_select(0, indices_i)
                tail_logit_i = self._compute_logit(hidden_i, weight_i,
                    bias_i, proj_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1,
                    target_i[:, None]).squeeze(1)
            if self.keep_order or keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
            offset += logprob_i.size(0)
    return nll
