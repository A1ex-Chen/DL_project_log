def convert(comp_type: str, src_file: Path, dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    dpr_state = DPRState.from_type(comp_type, src_file=src_file)
    model = dpr_state.load_dpr_model()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)
