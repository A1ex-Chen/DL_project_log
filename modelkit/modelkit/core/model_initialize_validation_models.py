def initialize_validation_models(self):
    try:
        generic_aliases = [t for t in self.__orig_bases__ if isinstance(t,
            typing._GenericAlias) and issubclass(t.__origin__, AbstractModel)]
        if len(generic_aliases):
            _item_type, _return_type = generic_aliases[0].__args__
            if _item_type != ItemType:
                self._item_type = _item_type
                type_name = self.__class__.__name__ + 'ItemTypeModel'
                self._item_model = pydantic.create_model(type_name, data=(
                    self._item_type, ...), __base__=InternalDataModel)
            if _return_type != ReturnType:
                self._return_type = _return_type
                type_name = self.__class__.__name__ + 'ReturnTypeModel'
                self._return_model = pydantic.create_model(type_name, data=
                    (self._return_type, ...), __base__=InternalDataModel)
    except Exception as exc:
        raise errors.ValidationInitializationException(
            f'{self.__class__.__name__}[{self.configuration_key}]',
            pydantic_exc=exc) from exc
