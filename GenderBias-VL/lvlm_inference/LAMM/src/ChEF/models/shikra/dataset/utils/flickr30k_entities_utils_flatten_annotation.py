def flatten_annotation(annotation_dir, indexes):
    data = []
    for index in tqdm(indexes):
        image_id = index
        ann_path = get_ann_path(index, annotation_dir=annotation_dir)
        sen_path = get_sen_path(index, annotation_dir=annotation_dir)
        anns = get_annotations(ann_path)
        sens = get_sentence_data(sen_path)
        for sen in sens:
            pids = list(set(phrase['phrase_id'] for phrase in sen['phrases'
                ] if phrase['phrase_id'] in anns['boxes']))
            boxes_mapping: Dict[str, List[int]] = {}
            boxes_filtered: List[List[int]] = []
            for pid in pids:
                v = anns['boxes'][pid]
                mapping = []
                for box in v:
                    mapping.append(len(boxes_filtered))
                    boxes_filtered.append(box)
                boxes_mapping[pid] = mapping
            boxes_seq: List[List[int]] = []
            for phrase in sen['phrases']:
                if not phrase['phrase_id'] in anns['boxes']:
                    continue
                pid = phrase['phrase_id']
                boxes_seq.append(boxes_mapping[pid])
            sent = list(sen['sentence'].split())
            for phrase in sen['phrases'][::-1]:
                if not phrase['phrase_id'] in anns['boxes']:
                    continue
                span = [phrase['first_word_index'], phrase[
                    'first_word_index'] + len(phrase['phrase'].split())]
                sent[span[0]:span[1]] = [
                    f"{PHRASE_ST_PLACEHOLDER}{' '.join(sent[span[0]:span[1]])}{PHRASE_ED_PLACEHOLDER}"
                    ]
            sent_converted = ' '.join(sent)
            assert len(re.findall(PHRASE_ST_PLACEHOLDER, sent_converted)
                ) == len(re.findall(PHRASE_ED_PLACEHOLDER, sent_converted)
                ) == len(boxes_seq
                ), f'error when parse: {sent_converted}, {boxes_seq}, {sen}, {anns}'
            assert sent_converted.replace(PHRASE_ST_PLACEHOLDER, '').replace(
                PHRASE_ED_PLACEHOLDER, '') == sen['sentence']
            item = {'id': len(data), 'image_id': image_id, 'boxes':
                boxes_filtered, 'sentence': sent_converted, 'boxes_seq':
                boxes_seq}
            data.append(item)
    return data
