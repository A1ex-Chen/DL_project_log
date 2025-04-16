@classmethod
@abc.abstractmethod
def get_update_cli_params(cls, **kwargs) ->typing.Dict[str, typing.Any]:
    """
            returns {
                "display":
                "params" :
            }
        where:
            - display is the versioning system specific text that will be display in the
            update cli
            - params are the params that will be forward to increment_version method

        """
    ...
