def batch_inference(self, model, batch, dataset, **kwargs):
    predictions = []
    prompts = self.instruction_handler.generate_basic_query(batch)
    sys_msg = None if not hasattr(dataset, 'system_msg'
        ) else dataset.system_msg
    outputs = model.batch_generate_3d(batch, prompts, sys_msg=sys_msg,
        dataset_name=dataset.dataset_name)
    for i in range(len(outputs)):
        answer_dict = {}
        answer_dict['query'] = prompts[i]
        answer_dict['answer'] = outputs[i]
        answer_dict['scene_id'] = batch['scene_id'][i]
        answer_dict['gt'] = batch['gt'][i]
        answer_dict['object_name'] = batch['object_name'][i]
        predictions.append(answer_dict)
    return predictions
