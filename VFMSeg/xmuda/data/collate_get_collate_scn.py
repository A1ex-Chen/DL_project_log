def get_collate_scn(is_train, with_vfm):
    return partial(collate_scn_base, output_orig=not is_train, with_vfm=
        with_vfm)
