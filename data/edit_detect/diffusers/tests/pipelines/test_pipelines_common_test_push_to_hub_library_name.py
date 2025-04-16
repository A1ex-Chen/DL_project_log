@unittest.skipIf(not is_jinja_available(), reason=
    'Model card tests cannot be performed without Jinja installed.')
def test_push_to_hub_library_name(self):
    components = self.get_pipeline_components()
    pipeline = StableDiffusionPipeline(**components)
    pipeline.push_to_hub(self.repo_id, token=TOKEN)
    model_card = ModelCard.load(f'{USER}/{self.repo_id}', token=TOKEN).data
    assert model_card.library_name == 'diffusers'
    delete_repo(self.repo_id, token=TOKEN)
