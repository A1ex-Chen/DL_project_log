def group_entities(self, entities: List[dict]) ->List[dict]:
    """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
    entity_groups = []
    entity_group_disagg = []
    if entities:
        last_idx = entities[-1]['index']
    for entity in entities:
        is_last_idx = entity['index'] == last_idx
        is_subword = self.ignore_subwords and entity['is_subword']
        if not entity_group_disagg:
            entity_group_disagg += [entity]
            if is_last_idx:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
            continue
        if (entity['entity'].split('-')[-1] == entity_group_disagg[-1][
            'entity'].split('-')[-1] and entity['entity'].split('-')[0] != 'B'
            ) and entity['index'] == entity_group_disagg[-1]['index'
            ] + 1 or is_subword:
            if is_subword:
                entity['entity'] = entity_group_disagg[-1]['entity'].split('-'
                    )[-1]
                entity['score'] = np.nan
            entity_group_disagg += [entity]
            if is_last_idx:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
        else:
            entity_groups += [self.group_sub_entities(entity_group_disagg)]
            entity_group_disagg = [entity]
            if is_last_idx:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
    return entity_groups
