def get_occ_features(job_list):
    gender_occ_img_features = {}
    male_features = []
    female_features = []
    genders = ['female', 'male']
    with torch.no_grad():
        for occ in tqdm(job_list):
            occ_name = '_'.join(occ.split(' '))
            occ_dir = os.path.join(test_root, occ_name)
            for gender in genders:
                sub_dir = os.path.join(occ_dir, gender)
                img_list = load_img_dirs(sub_dir)
                if img_list is None:
                    mean_img_features = torch.zeros(mean_img_features.shape
                        ).to(device)
                else:
                    _img_features = model.encode_image(img_list)
                    mean_img_features = _img_features.mean(dim=0)
                    del _img_features
                    mean_img_features = (mean_img_features /
                        mean_img_features.norm(dim=0, keepdim=True))
                gender_occ_img_features[occ, gender] = mean_img_features
                if gender == 'female':
                    female_features.append(mean_img_features)
                else:
                    male_features.append(mean_img_features)
                del img_list
        female_features = torch.stack(female_features, dim=0)
        male_features = torch.stack(male_features, dim=0)
    return gender_occ_img_features, female_features, male_features
