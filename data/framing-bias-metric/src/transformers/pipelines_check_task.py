def check_task(task: str) ->Tuple[Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`
            - :obj:`"sentiment-analysis"`
            - :obj:`"ner"`
            - :obj:`"question-answering"`
            - :obj:`"fill-mask"`
            - :obj:`"summarization"`
            - :obj:`"translation_xx_to_yy"`
            - :obj:`"translation"`
            - :obj:`"text-generation"`
            - :obj:`"conversational"`

    Returns:
        (task_defaults:obj:`dict`, task_options: (:obj:`tuple`, None)) The actual dictionary required to initialize the
        pipeline and some extra task options for parametrized tasks like "translation_XX_to_YY"


    """
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task, None
    if task.startswith('translation'):
        tokens = task.split('_')
        if len(tokens) == 4 and tokens[0] == 'translation' and tokens[2
            ] == 'to':
            targeted_task = SUPPORTED_TASKS['translation']
            return targeted_task, (tokens[1], tokens[3])
        raise KeyError(
            "Invalid translation task {}, use 'translation_XX_to_YY' format"
            .format(task))
    raise KeyError('Unknown task {}, available tasks are {}'.format(task, 
        list(SUPPORTED_TASKS.keys()) + ['translation_XX_to_YY']))
