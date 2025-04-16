def normalize(self, item):
    if isinstance(item, SquadExample):
        return item
    elif isinstance(item, dict):
        for k in ['question', 'context']:
            if k not in item:
                raise KeyError(
                    'You need to provide a dictionary with keys {question:..., context:...}'
                    )
            elif item[k] is None:
                raise ValueError('`{}` cannot be None'.format(k))
            elif isinstance(item[k], str) and len(item[k]) == 0:
                raise ValueError('`{}` cannot be empty'.format(k))
        return QuestionAnsweringPipeline.create_sample(**item)
    raise ValueError('{} argument needs to be of type (SquadExample, dict)'
        .format(item))
