def write(self, **kwargs):
    """
        Writes named list of dictionaries of np.ndarrays.
        Finally keyword names will be later prefixes of npz files where those dictionaries will be stored.

        ex. writer.write(inputs={'input': np.zeros((2, 10))},
                         outputs={'classes': np.zeros((2,)), 'probabilities': np.zeros((2, 32))},
                         labels={'classes': np.zeros((2,))})
        Args:
            **kwargs: named list of dictionaries of np.ndarrays to store
        """
    for prefix, data in kwargs.items():
        self._append_to_cache(prefix, data)
    biggest_item_size = max(self.cache_size.values())
    if biggest_item_size > self._flush_threshold_b:
        self.flush()
