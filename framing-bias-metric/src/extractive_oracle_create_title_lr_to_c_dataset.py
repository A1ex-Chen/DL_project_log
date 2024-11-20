def create_title_lr_to_c_dataset(allsides_title_triples):
    allsides_train, allsides_not_train = train_test_split(
        allsides_title_triples, test_size=0.1, random_state=42)
    allsides_test, allsides_val = train_test_split(allsides_not_train,
        test_size=0.66, random_state=42)
    for phase_triples, phase in [(allsides_test, 'test'), (allsides_val,
        'val'), (allsides_train, 'train')]:
        print(phase)
        data_name = 'title_lr_to_c'
        target_path = ('/home/nayeon/omission/data/aux_gen_task/{}/{}.target'
            .format(data_name, phase))
        source_path = ('/home/nayeon/omission/data/aux_gen_task/{}/{}.source'
            .format(data_name, phase))
        all_left = [triple['left'] for triple in phase_triples]
        all_right = [triple['right'] for triple in phase_triples]
        all_center = [triple['center'] for triple in phase_triples]
        for left, right, center in tqdm(zip(all_left, all_right, all_center
            ), total=len(all_left)):
            if len(center.split(' ')) > 4:
                source = '{} [SEP] {}'.format(right, left)
                target = center
                with open(source_path, 'a') as source_file:
                    source_file.write(source)
                    source_file.write('\n')
                with open(target_path, 'a') as target_file:
                    target_file.write(target)
                    target_file.write('\n')
