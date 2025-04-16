def build_nn(self):
    args = self.args
    device = self.device
    self.ae_training_kwarg = {'ae_loss_func': 'mse', 'ae_opt': 'sgd',
        'ae_lr': 0.2, 'ae_reg': 1e-05, 'lr_decay_factor': 1.0,
        'max_num_epochs': 1000, 'early_stop_patience': 50}
    self.encoder_kwarg = {'model_folder': './models/', 'data_root':
        DATA_ROOT, 'autoencoder_init': args.autoencoder_init,
        'training_kwarg': self.ae_training_kwarg, 'device': device,
        'verbose': True, 'rand_state': args.rng_seed}
    self.gene_encoder = get_gene_encoder(rnaseq_feature_usage=args.
        rnaseq_feature_usage, rnaseq_scaling=args.rnaseq_scaling, layer_dim
        =args.gene_layer_dim, num_layers=args.gene_num_layers, latent_dim=
        args.gene_latent_dim, **self.encoder_kwarg)
    self.drug_encoder = get_drug_encoder(drug_feature_usage=args.
        drug_feature_usage, dscptr_scaling=args.dscptr_scaling,
        dscptr_nan_threshold=args.dscptr_nan_threshold, layer_dim=args.
        drug_layer_dim, num_layers=args.drug_num_layers, latent_dim=args.
        drug_latent_dim, **self.encoder_kwarg)
    self.resp_net = RespNet(gene_latent_dim=args.gene_latent_dim,
        drug_latent_dim=args.drug_latent_dim, gene_encoder=self.
        gene_encoder, drug_encoder=self.drug_encoder, resp_layer_dim=args.
        resp_layer_dim, resp_num_layers_per_block=args.
        resp_num_layers_per_block, resp_num_blocks=args.resp_num_blocks,
        resp_num_layers=args.resp_num_layers, resp_dropout=args.dropout,
        resp_activation=args.resp_activation).to(device)
    print(self.resp_net)
    self.cl_clf_net_kwargs = {'encoder': self.gene_encoder, 'input_dim':
        args.gene_latent_dim, 'condition_dim': len(get_label_dict(DATA_ROOT,
        'data_src_dict.txt')), 'layer_dim': args.cl_clf_layer_dim,
        'num_layers': args.cl_clf_num_layers}
    self.category_clf_net = ClfNet(num_classes=len(get_label_dict(DATA_ROOT,
        'category_dict.txt')), **self.cl_clf_net_kwargs).to(device)
    self.site_clf_net = ClfNet(num_classes=len(get_label_dict(DATA_ROOT,
        'site_dict.txt')), **self.cl_clf_net_kwargs).to(device)
    self.type_clf_net = ClfNet(num_classes=len(get_label_dict(DATA_ROOT,
        'type_dict.txt')), **self.cl_clf_net_kwargs).to(device)
    self.drug_target_net = ClfNet(encoder=self.drug_encoder, input_dim=args
        .drug_latent_dim, condition_dim=0, layer_dim=args.
        drug_target_layer_dim, num_layers=args.drug_target_num_layers,
        num_classes=len(get_label_dict(DATA_ROOT, 'drug_target_dict.txt'))).to(
        device)
    self.drug_qed_net = RgsNet(encoder=self.drug_encoder, input_dim=args.
        drug_latent_dim, condition_dim=0, layer_dim=args.drug_qed_layer_dim,
        num_layers=args.drug_qed_num_layers, activation=args.
        drug_qed_activation).to(device)
    if args.multi_gpu:
        self.resp_net = nn.DataParallel(self.resp_net)
        self.category_clf_net = nn.DataParallel(self.category_clf_net)
        self.site_clf_net = nn.DataParallel(self.site_clf_net)
        self.type_clf_net = nn.DataParallel(self.type_clf_net)
        self.drug_target_net = nn.DataParallel(self.drug_target_net)
        self.drug_qed_net = nn.DataParallel(self.drug_qed_net)
