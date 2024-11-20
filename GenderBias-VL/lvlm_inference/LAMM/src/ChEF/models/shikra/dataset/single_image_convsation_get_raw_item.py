def get_raw_item(self, index) ->Dict[str, Any]:
    self.initialize_if_needed()
    return self.dataset[index]
