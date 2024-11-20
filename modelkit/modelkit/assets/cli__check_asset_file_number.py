def _check_asset_file_number(asset_path):
    n_files = len([f for f in glob.iglob(os.path.join(asset_path, '**/*'),
        recursive=True)])
    if n_files > 50:
        click.secho(
            f"""It looks like you are attempting to push an asset with more than 50 files in it ({n_files}).
This can lead to poor performance when retrieving the asset, and should be avoided.
You should consider archiving and compressing it."""
            , fg='red')
        if click.confirm('Proceed anyways ?', abort=True):
            pass
