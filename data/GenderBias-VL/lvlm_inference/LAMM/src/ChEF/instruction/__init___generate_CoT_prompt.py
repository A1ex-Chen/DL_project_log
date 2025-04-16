def generate_CoT_prompt(self, model, batch, max_new_tokens=256):
    cur_batch_len = get_cur_batch_len(batch)
    if 'question' in batch:
        question_list = batch['question']
        query_for_CoT = [f'{question}\n{self.CoT_prompt}' for question in
            question_list]
        CoT_response = model.batch_generate(batch['image_path'],
            query_for_CoT, max_new_tokens=max_new_tokens)
        query_for_answer = [self._query_format(self.prompt, question) for
            question in question_list]
    else:
        print(
            'You are using CoT inferencer for neither VQA tasks nor predefined prompt. It is not recommanded.'
            )
        query_for_CoT = [f'{self.prompt}\n{self.CoT_prompt}' for i in range
            (cur_batch_len)]
        CoT_response = model.batch_generate(batch['image_path'],
            query_for_CoT, max_new_tokens=max_new_tokens)
        query_for_answer = [f'{self.prompt}' for i in range(cur_batch_len)]
    return query_for_answer, CoT_response
