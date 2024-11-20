def format_prompt(instruction, input=None):
    PROMPT_DICT = {'prompt_input':
        """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
        , 'prompt_no_input':
        """Below is an instruction that describes a task. Write a response that appropriately completes the request.

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
