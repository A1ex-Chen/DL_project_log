def test_push_to_hub_in_organization(self):
    model = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32)
    model.push_to_hub(self.org_repo_id, token=TOKEN)
    new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        self.assertTrue(torch.equal(p1, p2))
    delete_repo(token=TOKEN, repo_id=self.org_repo_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN,
            repo_id=self.org_repo_id)
    new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        self.assertTrue(torch.equal(p1, p2))
    delete_repo(self.org_repo_id, token=TOKEN)
