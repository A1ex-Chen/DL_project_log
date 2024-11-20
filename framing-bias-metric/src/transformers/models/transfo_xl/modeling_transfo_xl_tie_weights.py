def tie_weights(self):
    """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """
    if self.config.tie_word_embeddings:
        for i in range(len(self.crit.out_layers)):
            self._tie_or_clone_weights(self.crit.out_layers[i], self.
                transformer.word_emb.emb_layers[i])
    if self.config.tie_projs:
        for i, tie_proj in enumerate(self.config.tie_projs):
            if (tie_proj and self.config.div_val == 1 and self.config.
                d_model != self.config.d_embed):
                if self.config.torchscript:
                    self.crit.out_projs[i] = nn.Parameter(self.transformer.
                        word_emb.emb_projs[0].clone())
                else:
                    self.crit.out_projs[i
                        ] = self.transformer.word_emb.emb_projs[0]
            elif tie_proj and self.config.div_val != 1:
                if self.config.torchscript:
                    self.crit.out_projs[i] = nn.Parameter(self.transformer.
                        word_emb.emb_projs[i].clone())
                else:
                    self.crit.out_projs[i
                        ] = self.transformer.word_emb.emb_projs[i]
