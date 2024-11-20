def group_sub_entities(self, entities: List[dict]) ->dict:
    """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
    entity = entities[0]['entity'].split('-')[-1]
    scores = np.nanmean([entity['score'] for entity in entities])
    tokens = [entity['word'] for entity in entities]
    entity_group = {'entity_group': entity, 'score': np.mean(scores),
        'word': self.tokenizer.convert_tokens_to_string(tokens), 'start':
        entities[0]['start'], 'end': entities[-1]['end']}
    return entity_group
