@property
def inputs(self) ->Mapping[str, Mapping[int, str]]:
    return OrderedDict([('input_ids', {(0): 'batch', (1): 'sequence'}), (
        'attention_mask', {(0): 'batch', (1): 'sequence'}), (
        'token_type_ids', {(0): 'batch', (1): 'sequence'})])
