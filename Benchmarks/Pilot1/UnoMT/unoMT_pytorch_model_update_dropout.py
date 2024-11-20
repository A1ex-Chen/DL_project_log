def update_dropout(self, dropout_rate):
    self.args.dropout = dropout_rate
    self.resp_net = RespNet(gene_latent_dim=self.args.gene_latent_dim,
        drug_latent_dim=self.args.drug_latent_dim, gene_encoder=self.
        gene_encoder, drug_encoder=self.drug_encoder, resp_layer_dim=self.
        args.resp_layer_dim, resp_num_layers_per_block=self.args.
        resp_num_layers_per_block, resp_num_blocks=self.args.
        resp_num_blocks, resp_num_layers=self.args.resp_num_layers,
        resp_dropout=self.args.dropout, resp_activation=self.args.
        resp_activation).to(self.device)
