@staticmethod
def create_sample(question: Union[str, List[str]], context: Union[str, List
    [str]]) ->Union[SquadExample, List[SquadExample]]:
    """
        QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally. This helper method
        encapsulate all the logic for converting question(s) and context(s) to :class:`~transformers.SquadExample`.

        We currently support extractive question answering.

        Arguments:
            question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
            context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of :class:`~transformers.SquadExample`: The corresponding :class:`~transformers.SquadExample`
            grouping question and context.
        """
    if isinstance(question, list):
        return [SquadExample(None, q, c, None, None, None) for q, c in zip(
            question, context)]
    else:
        return SquadExample(None, question, context, None, None, None)
