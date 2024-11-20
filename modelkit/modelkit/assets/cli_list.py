@assets_cli.command('list')
@click.option('--storage-prefix', envvar='MODELKIT_STORAGE_PREFIX')
def list(storage_prefix):
    """lists all available assets and their versions."""
    manager = StorageProvider(prefix=storage_prefix)
    console = Console()
    tree = Tree('[bold]Assets store[/bold]')
    tree.add(f'[dim]storage provider[/dim] {manager.driver.__class__.__name__}'
        )
    tree.add(f'[dim]prefix[/dim] {storage_prefix}')
    console.print(tree)
    table = Table(show_header=True, header_style='bold')
    table.add_column('Asset name')
    table.add_column('Versions', style='dim')
    n = 0
    n_versions = 0
    with Progress(SpinnerColumn(),
        '[progress.description]{task.description}', transient=True
        ) as progress:
        progress.add_task('Listing remote assets', start=False)
        for asset_name, versions_list in manager.iterate_assets():
            table.add_row(asset_name, ' '.join(versions_list))
            n += 1
            n_versions += len(versions_list)
    console.print(table)
    console.print(f'Found {n} assets ({n_versions} different versions)')
