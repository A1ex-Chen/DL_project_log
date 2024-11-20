def __getitem__(self, index) ->Dict[str, str]:
    index = index + 1
    source_line = self.prefix + linecache.getline(str(self.src_file), index
        ).rstrip('\n')
    tgt_line = linecache.getline(str(self.tgt_file), index).rstrip('\n')
    assert source_line, f'empty source line for index {index}'
    assert tgt_line, f'empty tgt line for index {index}'
    returning_obj = {'tgt_texts': tgt_line, 'src_texts': source_line, 'id':
        index - 1}
    if self.mt:
        if self.extra_task in CLASSIFICATION_TASKS:
            idx = index % len(self.extra_task_data)
            obj = self.extra_task_data[idx]
            extra_task_text = obj['text'][:1000]
            extra_task_label = self.label_map[obj['label']]
            returning_obj['extra_task_text'] = extra_task_text
            returning_obj['extra_task_label'] = extra_task_label
        elif self.extra_task in GENERATION_TASKS:
            extra_source_line = self.prefix + linecache.getline(str(self.
                extra_src_file), index).rstrip('\n')
            extra_tgt_line = linecache.getline(str(self.extra_tgt_file), index
                ).rstrip('\n')
            returning_obj['extra_tgt_texts'] = extra_tgt_line
            returning_obj['extra_src_texts'] = extra_source_line
    return returning_obj
