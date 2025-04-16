def clean_main_ref_in_model_list():
    """Replace the links from main doc tp stable doc in the model list of the README."""
    _start_prompt = (
        '🤗 Transformers currently provides the following architectures')
    _end_prompt = '1. Want to contribute a new model?'
    with open(README_FILE, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    start_index = 0
    while not lines[start_index].startswith(_start_prompt):
        start_index += 1
    start_index += 1
    index = start_index
    while not lines[index].startswith(_end_prompt):
        if lines[index].startswith('1.'):
            lines[index] = lines[index].replace(
                'https://huggingface.co/docs/diffusers/main/model_doc',
                'https://huggingface.co/docs/diffusers/model_doc')
        index += 1
    with open(README_FILE, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lines)
