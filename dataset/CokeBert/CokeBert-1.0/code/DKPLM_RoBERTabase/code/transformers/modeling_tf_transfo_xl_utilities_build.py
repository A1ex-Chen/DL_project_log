def build(self, input_shape):
    if self.n_clusters > 0:
        self.cluster_weight = self.add_weight(shape=(self.n_clusters, self.
            d_embed), initializer='zeros', trainable=True, name=
            'cluster_weight')
        self.cluster_bias = self.add_weight(shape=(self.n_clusters,),
            initializer='zeros', trainable=True, name='cluster_bias')
    if self.div_val == 1:
        for i in range(len(self.cutoffs)):
            if self.d_proj != self.d_embed:
                weight = self.add_weight(shape=(self.d_embed, self.d_proj),
                    initializer='zeros', trainable=True, name=
                    'out_projs_._{}'.format(i))
                self.out_projs.append(weight)
            else:
                self.out_projs.append(None)
            weight = self.add_weight(shape=(self.n_token, self.d_embed),
                initializer='zeros', trainable=True, name=
                'out_layers_._{}_._weight'.format(i))
            bias = self.add_weight(shape=(self.n_token,), initializer=
                'zeros', trainable=True, name='out_layers_._{}_._bias'.
                format(i))
            self.out_layers.append((weight, bias))
    else:
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = self.d_embed // self.div_val ** i
            weight = self.add_weight(shape=(d_emb_i, self.d_proj),
                initializer='zeros', trainable=True, name='out_projs_._{}'.
                format(i))
            self.out_projs.append(weight)
            weight = self.add_weight(shape=(r_idx - l_idx, d_emb_i),
                initializer='zeros', trainable=True, name=
                'out_layers_._{}_._weight'.format(i))
            bias = self.add_weight(shape=(r_idx - l_idx,), initializer=
                'zeros', trainable=True, name='out_layers_._{}_._bias'.
                format(i))
            self.out_layers.append((weight, bias))
    super(TFAdaptiveSoftmaxMask, self).build(input_shape)
