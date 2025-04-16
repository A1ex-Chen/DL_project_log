def update(self, config_dict: Dict[str, Any]):
    """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that shall be updated for this class.
        """
    for key, value in config_dict.items():
        setattr(self, key, value)
