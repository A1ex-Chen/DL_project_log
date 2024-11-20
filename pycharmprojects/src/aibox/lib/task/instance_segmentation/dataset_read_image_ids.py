def read_image_ids(path_to_split_txt: str) ->List[str]:
    with open(path_to_split_txt, 'r') as f:
        lines = f.readlines()
        return [os.path.splitext(line.rstrip())[0] for line in lines]
