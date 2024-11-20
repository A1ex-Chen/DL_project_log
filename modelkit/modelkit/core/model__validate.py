def _validate(self, item: Any, model: Union[Type[InternalDataModel], None],
    exception: Type[errors.ModelkitDataValidationException]):
    if model:
        try:
            return model(data=item).data
        except pydantic.ValidationError as exc:
            raise exception(
                f'{self.__class__.__name__}[{self.configuration_key}]',
                pydantic_exc=exc) from exc
    return item
