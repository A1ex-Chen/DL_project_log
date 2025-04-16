def forward(self, hidden, labels=None, keep_order=False):
    """
            Params:
                hidden :: [len*bsz x d_proj]
                labels :: [len*bsz]
            Return:
                if labels is None:
                    out :: [len*bsz] Negative log likelihood
                else:
                    out :: [len*bsz x n_tokens] log probabilities of tokens over the vocabulary
            We could replace this implementation by the native PyTorch one
            if their's had an option to set bias on all clusters in the native one.
            here: https://github.com/pytorch/pytorch/blob/dbe6a7a9ff1a364a8706bf5df58a1ca96d2fd9da/torch/nn/modules/adaptive.py#L138
        """
    if labels is not None:
        labels = labels.view(-1)
        if hidden.size(0) != labels.size(0):
            raise RuntimeError(
                'Input and labels should have the same size in the batch dimension.'
                )
    if self.n_clusters == 0:
        logit = self._compute_logit(hidden, self.out_layers[0].weight, self
            .out_layers[0].bias, self.out_projs[0])
        if labels is not None:
            out = -F.log_softmax(logit, dim=-1).gather(1, labels.unsqueeze(1)
                ).squeeze(1)
        else:
            out = F.log_softmax(logit, dim=-1)
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
        head_logprob = F.log_softmax(head_logit, dim=1)
        if labels is None:
            out = hidden.new_empty((head_logit.size(0), self.n_token))
        else:
            out = torch.zeros_like(labels, dtype=hidden.dtype, device=
                hidden.device)
        offset = 0
        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
            if labels is not None:
                mask_i = (labels >= l_idx) & (labels < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                target_i = labels.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)
                hidden_i = hidden.index_select(0, indices_i)
            else:
                hidden_i = hidden
            if i == 0:
                if labels is not None:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]
                        ).squeeze(1)
                else:
                    out[:, :self.cutoffs[0]] = head_logprob[:, :self.cutoffs[0]
                        ]
            else:
                weight_i, bias_i, proj_i = weights[i], biases[i
                    ], self.out_projs[i]
                tail_logit_i = self._compute_logit(hidden_i, weight_i,
                    bias_i, proj_i)
                tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                cluster_prob_idx = self.cutoffs[0] + i - 1
                if labels is not None:
                    logprob_i = head_logprob_i[:, cluster_prob_idx
                        ] + tail_logprob_i.gather(1, target_i[:, None]
                        ).squeeze(1)
                else:
                    logprob_i = head_logprob[:, cluster_prob_idx, None
                        ] + tail_logprob_i
                    out[:, l_idx:r_idx] = logprob_i
            if labels is not None:
                if hasattr(self, 'keep_order'
                    ) and self.keep_order or keep_order:
                    out.index_copy_(0, indices_i, -logprob_i)
                else:
                    out[offset:offset + logprob_i.size(0)].copy_(-logprob_i)
                offset += logprob_i.size(0)
    return out
