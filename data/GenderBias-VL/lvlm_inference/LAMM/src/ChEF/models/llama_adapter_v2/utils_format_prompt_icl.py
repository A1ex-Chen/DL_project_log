def format_prompt_icl(instruction, input=None):
    PROMPT_DICT = {'prompt_input':
        """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.You will now see some examples. The example has no relation to the provided image content. You need to follow the example and answer the final question based on the image content.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
        , 'prompt_no_input':
        """Below is an instruction that describes a task. Write a response that appropriately completes the request.You will now see some examples. The example has no relation to the provided image content. You need to follow the example and answer the final question based on the image content.

### Instruction:
{instruction}

### Response:"""
        }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction':
            instruction})
    else:
        return PROMPT_DICT['prompt_input'].format_map({'instruction':
            instruction, 'input': input})
