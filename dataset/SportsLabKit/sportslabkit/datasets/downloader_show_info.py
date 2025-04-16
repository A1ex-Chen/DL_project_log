def show_info(self) ->None:
    """Show the dataset info."""
    dataset = self.api.dataset_list(search=
        f'{self.dataset_owner}/{self.dataset_name}')[0]
    inspect(dataset)
