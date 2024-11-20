def exclusive_group(group, name, default, help):
    destname = name.replace('-', '_')
    subgroup = group.add_mutually_exclusive_group(required=False)
    subgroup.add_argument(f'--{name}', dest=f'{destname}', action=
        'store_true', help=f"{help} (use '--no-{name}' to disable)")
    subgroup.add_argument(f'--no-{name}', dest=f'{destname}', action=
        'store_false', help=argparse.SUPPRESS)
    subgroup.set_defaults(**{destname: default})
