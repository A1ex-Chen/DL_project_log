def test_smart_download(self):
    model_id = 'hf-internal-testing/unet-pipeline-dummy'
    with tempfile.TemporaryDirectory() as tmpdirname:
        _ = DiffusionPipeline.from_pretrained(model_id, cache_dir=
            tmpdirname, force_download=True)
        local_repo_name = '--'.join(['models'] + model_id.split('/'))
        snapshot_dir = os.path.join(tmpdirname, local_repo_name, 'snapshots')
        snapshot_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])
        assert os.path.isfile(os.path.join(snapshot_dir, DiffusionPipeline.
            config_name))
        assert os.path.isfile(os.path.join(snapshot_dir, CONFIG_NAME))
        assert os.path.isfile(os.path.join(snapshot_dir, SCHEDULER_CONFIG_NAME)
            )
        assert os.path.isfile(os.path.join(snapshot_dir, WEIGHTS_NAME))
        assert os.path.isfile(os.path.join(snapshot_dir, 'scheduler',
            SCHEDULER_CONFIG_NAME))
        assert os.path.isfile(os.path.join(snapshot_dir, 'unet', WEIGHTS_NAME))
        assert os.path.isfile(os.path.join(snapshot_dir, 'unet', WEIGHTS_NAME))
        assert not os.path.isfile(os.path.join(snapshot_dir, 'big_array.npy'))
