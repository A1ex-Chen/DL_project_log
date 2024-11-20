@unittest.skipIf(not is_jinja_available(), reason=
    'Model card tests cannot be performed without Jinja installed.')
def test_push_to_hub_library_name(self):
    model = UNet2DConditionModel(block_out_channels=(32, 64),
        layers_per_block=2, sample_size=32, in_channels=4, out_channels=4,
        down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
        cross_attention_dim=32)
    model.push_to_hub(self.repo_id, token=TOKEN)
    model_card = ModelCard.load(f'{USER}/{self.repo_id}', token=TOKEN).data
    assert model_card.library_name == 'diffusers'
    delete_repo(self.repo_id, token=TOKEN)
