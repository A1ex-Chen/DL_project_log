def ppl_answer_multi(self, image_list, question_list, chat_list,
    answer_list, answer_options, CoT_list=None, calib=False):
    embs_list = []
    for idx, (image, question, conv, answer) in enumerate(zip(image_list,
        question_list, chat_list, answer_list)):
        img_list = []
        self.upload_imgs(image, conv, img_list)
        self.ask(question, conv)
        conv.append_message(conv.roles[1], None)
        if CoT_list is not None:
            embs = self.get_context_emb(conv, img_list, answer=CoT_list[idx
                ] + answer)
        else:
            embs = self.get_context_emb(conv, img_list, answer=answer)
        embs_list.append(embs)
    results = self.do_ppl(embs_list, answer_list, answer_options, calib=calib)
    return results
