def load_url(self, url):
    """Load a module dictionary from url.

        Args:
            url (str): url to saved model
        """
    print(url)
    print('=> Loading checkpoint from url...')
    state_dict = model_zoo.load_url(url, progress=True)
    scalars = self.parse_state_dict(state_dict)
    return scalars
