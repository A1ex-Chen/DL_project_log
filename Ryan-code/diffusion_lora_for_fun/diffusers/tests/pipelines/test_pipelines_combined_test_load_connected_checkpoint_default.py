def test_load_connected_checkpoint_default(self):
    prior = KandinskyV22PriorPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-prior')
    decoder = KandinskyV22Pipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder')
    assert KandinskyV22CombinedPipeline._load_connected_pipes
    pipeline = KandinskyV22CombinedPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder')
    prior_comps = prior.components
    decoder_comps = decoder.components
    for k, component in pipeline.components.items():
        if k.startswith('prior_'):
            k = k[6:]
            comp = prior_comps[k]
        else:
            comp = decoder_comps[k]
        if isinstance(component, torch.nn.Module):
            assert state_dicts_almost_equal(component.state_dict(), comp.
                state_dict())
        elif hasattr(component, 'config'):
            assert dict(component.config) == dict(comp.config)
        else:
            assert component.__class__ == comp.__class__
