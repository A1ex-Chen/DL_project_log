def test_push_to_hub_in_organization(self):
    components = self.get_pipeline_components()
    pipeline = StableDiffusionPipeline(**components)
    pipeline.push_to_hub(self.org_repo_id, token=TOKEN)
    new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id,
        subfolder='unet')
    unet = components['unet']
    for p1, p2 in zip(unet.parameters(), new_model.parameters()):
        self.assertTrue(torch.equal(p1, p2))
    delete_repo(token=TOKEN, repo_id=self.org_repo_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN,
            repo_id=self.org_repo_id)
    new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id,
        subfolder='unet')
    for p1, p2 in zip(unet.parameters(), new_model.parameters()):
        self.assertTrue(torch.equal(p1, p2))
    delete_repo(self.org_repo_id, token=TOKEN)
