def create_left_source_2_intersection_target_dataset(allsides_triples):
    allsides_train, allsides_not_train = train_test_split(allsides_triples,
        test_size=0.1, random_state=42)
    allsides_test, allsides_val = train_test_split(allsides_not_train,
        test_size=0.66, random_state=42)
    for phase_triples, phase in [(allsides_test, 'test'), (allsides_val,
        'val'), (allsides_train, 'train')]:
        print(phase)
        data_name = 'l_to_intersect'
        target_path = ('/home/nayeon/omission/data/aux_gen_task/{}/{}.target'
            .format(data_name, phase))
        source_path = ('/home/nayeon/omission/data/aux_gen_task/{}/{}.source'
            .format(data_name, phase))
        all_left = [triple['left'] for triple in phase_triples]
        intersection_in_left = find_left_right_intersection('left',
            phase_triples)
        for source, target in tqdm(zip(all_left, intersection_in_left),
            total=len(intersection_in_left)):
            source = source.replace('\n', ' ')
            target = target.replace('\n', ' ')
            if len(intersection_in_left) > 1:
                with open(source_path, 'a') as source_file:
                    source_file.write(source)
                    source_file.write('\n')
                with open(target_path, 'a') as target_file:
                    target_file.write(target)
                    target_file.write('\n')
