@property
def pipeline_class(self) ->Union[Callable, DiffusionPipeline]:
    raise NotImplementedError(
        'You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. See existing pipeline tests for reference.'
        )
