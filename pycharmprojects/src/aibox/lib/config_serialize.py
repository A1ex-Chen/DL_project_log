def serialize(self, path_to_pickle_file: str):
    with open(path_to_pickle_file, 'wb') as f:
        pickle.dump(self, f)
