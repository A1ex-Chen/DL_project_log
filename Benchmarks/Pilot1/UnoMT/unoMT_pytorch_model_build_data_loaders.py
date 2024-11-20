def build_data_loaders(self):
    args = self.args
    self.drug_resp_trn_loader = torch.utils.data.DataLoader(DrugRespDataset
        (data_src=args.train_sources, training=True, **self.
        drug_resp_dataset_kwargs), batch_size=args.trn_batch_size, **self.
        dataloader_kwargs)
    self.drug_resp_val_loaders = [torch.utils.data.DataLoader(
        DrugRespDataset(data_src=src, training=False, **self.
        drug_resp_dataset_kwargs), batch_size=args.val_batch_size, **self.
        dataloader_kwargs) for src in args.val_sources]
    self.cl_clf_trn_loader = torch.utils.data.DataLoader(CLClassDataset(
        training=True, **self.cl_clf_dataset_kwargs), batch_size=args.
        trn_batch_size, **self.dataloader_kwargs)
    self.cl_clf_val_loader = torch.utils.data.DataLoader(CLClassDataset(
        training=False, **self.cl_clf_dataset_kwargs), batch_size=args.
        val_batch_size, **self.dataloader_kwargs)
    self.drug_target_trn_loader = torch.utils.data.DataLoader(DrugTargetDataset
        (training=True, **self.drug_target_dataset_kwargs), batch_size=args
        .trn_batch_size, **self.dataloader_kwargs)
    self.drug_target_val_loader = torch.utils.data.DataLoader(DrugTargetDataset
        (training=False, **self.drug_target_dataset_kwargs), batch_size=
        args.val_batch_size, **self.dataloader_kwargs)
    self.drug_qed_trn_loader = torch.utils.data.DataLoader(DrugQEDDataset(
        training=True, **self.drug_qed_dataset_kwargs), batch_size=args.
        trn_batch_size, **self.dataloader_kwargs)
    self.drug_qed_val_loader = torch.utils.data.DataLoader(DrugQEDDataset(
        training=False, **self.drug_qed_dataset_kwargs), batch_size=args.
        val_batch_size, **self.dataloader_kwargs)
