@property
def original_pipeline_class(self):
    if 'xl' in self.pipeline_class.__name__.lower():
        original_pipeline_class = StableDiffusionXLPipeline
    else:
        original_pipeline_class = StableDiffusionPipeline
    return original_pipeline_class
