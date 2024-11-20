def test_kwargs_local_files_only(self):
    repo = 'hf-internal-testing/tiny-stable-diffusion-torch'
    tmpdirname = DiffusionPipeline.download(repo)
    tmpdirname = Path(tmpdirname)
    commit_id = tmpdirname.name
    new_commit_id = commit_id + 'hug'
    ref_dir = tmpdirname.parent.parent / 'refs/main'
    with open(ref_dir, 'w') as f:
        f.write(new_commit_id)
    new_tmpdirname = tmpdirname.parent / new_commit_id
    os.rename(tmpdirname, new_tmpdirname)
    try:
        AutoPipelineForText2Image.from_pretrained(repo, local_files_only=True)
    except OSError:
        assert False, 'not able to load local files'
    shutil.rmtree(tmpdirname.parent.parent)
