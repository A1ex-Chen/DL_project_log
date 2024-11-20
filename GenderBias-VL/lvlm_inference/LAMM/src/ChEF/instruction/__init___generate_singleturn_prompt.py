def generate_singleturn_prompt(self, batch, batch_idx=0):
    prompt = self.prompt
    cur_batch_len = get_cur_batch_len(batch)
    question_list = batch['question'] if 'question' in batch else [''
        ] * cur_batch_len
    query = [self._query_format(prompt, question) for question in question_list
        ]
    if self.incontext_cfg:
        batch_ices = self.generate_ices(query, batch_idx, cur_batch_len)
        for idx, ices in enumerate(batch_ices):
            ice_image_path = [ice['image_path'] for ice in ices]
            query_image_path = batch['image_path'][idx]
            if isinstance(query_image_path, str):
                query_image_path = [query_image_path]
            batch['image_path'][idx] = ice_image_path + query_image_path
            ice_query = '\n'.join([(ice['question'] + '\n' + self.
                answer_template.format(option=ice['gt_answers'])) for ice in
                ices])
            query[idx] = ice_query + query[idx]
    return query
