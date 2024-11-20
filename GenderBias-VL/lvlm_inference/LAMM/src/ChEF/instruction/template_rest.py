# classification
coarse_grained_classification_answer_templates = [
    'The object in the image is {option}',
    '{option}',
    'The image features {option}',
    'The image shows {option}',
]
fine_grained_classification_answer_templates = [
    'The fine-grained category label for this image is {option}',
]

# caption
caption_answer_templates = [
    'The caption for this image is \" {option}',
    '{option}',
]

# vqa
vqa_answer_templates = [
    'The answer is {option}',
    'The correct option for the question is {option}',
    '{option}',
]

# counting
counting_answer_templates = ['{option}']

# detection
detection_answer_templates = [
    ['The image shows {option}', 'It is located at the bbox {option}'], # default
    ['The object in the image is {option}', 'The {option}'], # For shikra
    ['The object in the image is {option}', '{option}'] # For kosmos2
]

# POPE
pope_answer_templates=[ 
    '{option}',
    'The answer is {option}'
]

# octavius3d
octavius3d_answer_templates = ['{option}']

winoground_templates = ['{option}']

answer_template_dict = {
    'coarse_grained_classification_templates': coarse_grained_classification_answer_templates,
    'fine_grained_classification_templates': fine_grained_classification_answer_templates,
    'VQA_templates': vqa_answer_templates,
    'counting_templates': counting_answer_templates,
    'caption_templates': caption_answer_templates,
    'detection_templates': detection_answer_templates,
    'POPE_templates': pope_answer_templates,
    'Classification_octavius3d_templates': octavius3d_answer_templates,
    'Caption_octavius3d_templates': octavius3d_answer_templates,
    'VQA_octavius3d_templates': octavius3d_answer_templates,
    'Winoground_templates': winoground_templates,
}



if __name__ == '__main__':
    test_dict = {
        'coarse_grained_classification': ['singleturn'],
        'fine_grained_classification': ['singleturn', 'multiturn'],
        'VQA': ['singleturn'],
        'counting': ['singleturn'],
        'caption': ['singleturn'],
        'detection': ['singleturn', 'multiturn']
    }

    for key, value in test_dict.items():
        for prompt_type in value:
            template = build_template(task_name=key, assigned_ids=-1, prompt_type=prompt_type)
    
    for key, value in test_dict.items():
        for prompt_type in value:
            template = build_template(task_name=key, defined_template='Test template {option}.', prompt_type=prompt_type)