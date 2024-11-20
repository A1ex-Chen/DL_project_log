def update_l2regularizer(self, reg):
    args = self.args
    self.resp_opt = get_optimizer(opt_type=args.resp_opt, networks=self.
        resp_net, learning_rate=args.resp_lr, l2_regularization=reg)
    self.cl_clf_opt = get_optimizer(opt_type=args.cl_clf_opt, networks=[
        self.category_clf_net, self.site_clf_net, self.type_clf_net],
        learning_rate=self.args.cl_clf_lr, l2_regularization=reg)
    self.drug_target_opt = get_optimizer(opt_type=args.drug_target_opt,
        networks=self.drug_target_net, learning_rate=args.drug_target_lr,
        l2_regularization=reg)
    self.drug_qed_opt = get_optimizer(opt_type=args.drug_qed_opt, networks=
        self.drug_qed_net, learning_rate=args.drug_qed_lr,
        l2_regularization=reg)
