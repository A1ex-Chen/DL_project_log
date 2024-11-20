def __init__(self, **kwargs):
    self.model_details = kwargs.pop('model_details', {})
    self.intended_use = kwargs.pop('intended_use', {})
    self.factors = kwargs.pop('factors', {})
    self.metrics = kwargs.pop('metrics', {})
    self.evaluation_data = kwargs.pop('evaluation_data', {})
    self.training_data = kwargs.pop('training_data', {})
    self.quantitative_analyses = kwargs.pop('quantitative_analyses', {})
    self.ethical_considerations = kwargs.pop('ethical_considerations', {})
    self.caveats_and_recommendations = kwargs.pop('caveats_and_recommendations'
        , {})
    for key, value in kwargs.items():
        try:
            setattr(self, key, value)
        except AttributeError as err:
            logger.error("Can't set {} with value {} for {}".format(key,
                value, self))
            raise err
