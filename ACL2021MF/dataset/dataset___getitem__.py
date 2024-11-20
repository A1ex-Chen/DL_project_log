def __getitem__(self, index):
    (concept_set, pos, cls_on_input, concept_cls, copy_mention_flag,
        decoder_mention_flag, gen, gt, gt_concept) = self.record[index]
    item = {'gt': gt, 'gt_concepts': gt_concept, 'concept_set': concept_set,
        'copy_pos': pos, 'concept_cls': concept_cls, 'copy_mention_flag':
        copy_mention_flag, 'decoder_mention_flag': decoder_mention_flag,
        'cls_on_input': cls_on_input}
    if self.is_training:
        item['gen'] = gen
    return item
