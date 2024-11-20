def generate_images(self, start_index=0, end_index=1):
    each_occ_nums = 20
    each_occ_batch = 4
    n_steps = 50
    high_noise_frac = 0.8
    genders = ['female', 'male']
    each_occ_trial = 100
    edit_guidance_scale = 7.5
    img_height, img_width = 768, 768
    for occ in tqdm(self.all_occ_list[start_index:end_index]):
        occ_name = '_'.join(occ.split(' '))
        occ_dir = os.path.join(exp_dir, sub_exp, occ_name)
        cf_occ_dir = os.path.join(exp_dir, sub_cf_exp, occ_name)
        fail_occ_dir = os.path.join(fail_exp_dir, sub_exp, occ_name)
        fail_cf_occ_dir = os.path.join(fail_exp_dir, sub_cf_exp, occ_name)
        for gender in genders:
            output_dir = os.path.join(occ_dir, gender)
            os.makedirs(output_dir, exist_ok=True)
            cf_output_dir = os.path.join(cf_occ_dir, gender)
            os.makedirs(cf_output_dir, exist_ok=True)
            fail_output_dir = os.path.join(fail_occ_dir, gender)
            os.makedirs(fail_output_dir, exist_ok=True)
            fail_cf_output_dir = os.path.join(fail_cf_occ_dir, gender)
            os.makedirs(fail_cf_output_dir, exist_ok=True)
            prompts = self.occ_gender_prompts_map[occ, gender]
            cur_num, fail_cur_num = 0, 0
            trial = 0
            batch_i = 0
            while trial < each_occ_trial:
                prompt_id = batch_i % len(prompts)
                prompt = prompts[batch_i % len(prompts)]
                batch_i += 1
                print(f'occ: {occ}, gender: {gender}, prompt: {prompt}')
                prompt_list = [prompt] * each_occ_batch
                neg_prompt_list = self.neg_list * each_occ_batch
                image = self.base(prompt=prompt_list, negative_prompt=
                    neg_prompt_list, num_inference_steps=n_steps,
                    denoising_end=high_noise_frac, output_type='latent',
                    generator=self.generator, height=img_height, width=
                    img_width).images
                base_images = self.refiner(prompt=prompt_list,
                    negative_prompt=neg_prompt_list, num_inference_steps=
                    n_steps, denoising_start=high_noise_frac, image=image,
                    generator=self.generator).images
                cf_gender = self.cf_gender_map[gender]
                cf_prompt = (
                    f'turn {self.gender_reso[gender]} into a {cf_gender}')
                prompt_list = [cf_prompt] * each_occ_batch
                cf_images = self.edit_model(prompt=prompt_list, image=
                    base_images, edit_guidance_scale=edit_guidance_scale,
                    generator=self.generator).images
                test_labels = []
                for _gender in [gender, cf_gender]:
                    test_labels.append(f'{_gender}')
                answer_base = f'{gender}'
                answer_cf = f'{cf_gender}'
                test_label_emb = self.get_clip_text_emb(test_labels)
                base_img_emb = self.get_clip_image_emb(base_images)
                cf_img_emb = self.get_clip_image_emb(cf_images)
                score_base = (100.0 * base_img_emb @ test_label_emb.T).softmax(
                    dim=-1)
                score_cf = (100.0 * cf_img_emb @ test_label_emb.T).softmax(dim
                    =-1)
                for _ind in range(len(base_images)):
                    pred_base = test_labels[score_base[_ind].argmax()]
                    pred_cf = test_labels[score_cf[_ind].argmax()]
                    if pred_base == answer_base and pred_cf == answer_cf:
                        img_base = base_images[_ind]
                        img_base.save(
                            f'{output_dir}/{gender}_{occ_name}_{cur_num}.png')
                        img_cf = cf_images[_ind]
                        img_cf.save(
                            f'{cf_output_dir}/{gender}_{occ_name}_{cur_num}.png'
                            )
                        print(
                            f'{test_labels}\tbase: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}'
                            )
                        self.flog.write(
                            f"""{test_labels}	base: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}
"""
                            )
                        self.flog.flush()
                        self.flog_propmpt.write(
                            f"""{gender}_{occ_name}_{cur_num}.png,{prompt_id},{prompt}
"""
                            )
                        self.flog_propmpt.flush()
                        cur_num += 1
                        print('gender flip')
                    else:
                        img_base = base_images[_ind]
                        img_base.save(
                            f'{fail_output_dir}/{gender}_{occ_name}_{fail_cur_num}.png'
                            )
                        img_cf = cf_images[_ind]
                        img_cf.save(
                            f'{fail_cf_output_dir}/{gender}_{occ_name}_{fail_cur_num}.png'
                            )
                        print(
                            f'{test_labels}\tbase: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}'
                            )
                        self.fail_flog.write(
                            f"""{test_labels}	base: {score_base[_ind].tolist()}, cf: {score_cf[_ind].tolist()}
"""
                            )
                        self.fail_flog.flush()
                        self.fail_flog_propmpt.write(
                            f"""{gender}_{occ_name}_{cur_num}.png,{prompt_id},{prompt}
"""
                            )
                        self.fail_flog_propmpt.flush()
                        fail_cur_num += 1
                        print('gender fail')
                    trial += 1
                    print(
                        f'answer_base: {answer_base}, pred_base:{pred_base}, answer_cf: {answer_cf}, pred_cf:{pred_cf}'
                        )
