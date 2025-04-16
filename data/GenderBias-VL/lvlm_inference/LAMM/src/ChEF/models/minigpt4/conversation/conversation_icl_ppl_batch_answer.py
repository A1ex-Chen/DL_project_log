def icl_ppl_batch_answer(self, image_list, question_list, chat_list,
    answer_list, answer_options, ice_imgs_emb, sample_data, incontext_cfg,
    CoT_list=None):
    embs_list = []
    prompts = []
    for i, (image, question, conv, answer) in enumerate(zip(image_list,
        question_list, chat_list, answer_list)):
        img_list = []
        self.upload_img(image, conv, img_list)
        img_list = self.get_icl_prompt_img(question, conv, img_list,
            sample_data[i], ice_imgs_emb, i, incontext_cfg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
        if CoT_list is not None:
            embs = self.get_context_emb(conv, img_list, answer=CoT_list[i] +
                answer)
        else:
            embs = self.get_context_emb(conv, img_list, answer=answer)
        embs_list.append(embs)
    results = self.do_ppl(embs_list, answer_list, answer_options)
    return results, prompts
