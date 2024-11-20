@classmethod
@abc.abstractmethod
def increment_version(cls, version_list: typing.Optional[typing.List[str]]=
    None, params: typing.Optional[typing.Dict[str, str]]=None) ->str:
    """Algorithm used to increment your version. Returns new version"""
    ...
