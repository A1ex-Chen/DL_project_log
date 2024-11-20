@staticmethod
def from_type(comp_type: str, *args, **kwargs) ->'DPRState':
    if comp_type.startswith('c'):
        return DPRContextEncoderState(*args, **kwargs)
    if comp_type.startswith('q'):
        return DPRQuestionEncoderState(*args, **kwargs)
    if comp_type.startswith('r'):
        return DPRReaderState(*args, **kwargs)
    else:
        raise ValueError(
            "Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'."
            )
