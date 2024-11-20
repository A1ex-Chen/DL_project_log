def build_dataset(self):
    self.transform = T.Compose([T.RandomResize([800], max_size=1333), T.
        ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    self.base_dataset = BaseDataset(self.transform, root_path=self.args.
        base_root, occupation_path=self.args.occ_path)
    self.base_dataloader = DataLoader(self.base_dataset, batch_size=self.
        args.batch_size, shuffle=False, num_workers=4)
    self.cf_dataset = BaseDataset(self.transform, root_path=self.args.
        cf_root, occupation_path=self.args.occ_path)
    self.cf_dataloader = DataLoader(self.cf_dataset, batch_size=self.args.
        batch_size, shuffle=False, num_workers=4)
