@torch.no_grad()
def generate(self, images, texts, num_beams=1, max_new_tokens=20,
    min_length=1, top_p=0.9, repetition_penalty=1, length_penalty=1,
    temperature=1, do_sample=False, stop_words_ids=[2]):
    """
            function for generate test use
        """
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[
        torch.tensor([i]).to(self.device) for i in stop_words_ids])])
    img_embeds, atts_img = self.encode_img(images.to(self.device))
    image_lists = [[image_emb[None]] for image_emb in img_embeds]
    batch_embs = [self.get_context_emb(text, img_list) for text, img_list in
        zip(texts, image_lists)]
    batch_size = len(batch_embs)
    max_len = max([emb.shape[1] for emb in batch_embs])
    emb_dim = batch_embs[0].shape[2]
    dtype = batch_embs[0].dtype
    device = batch_embs[0].device
    embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=
        device)
    attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=
        device)
    for i, emb in enumerate(batch_embs):
        emb_len = emb.shape[1]
        embs[i, -emb_len:] = emb[0]
        attn_mask[i, -emb_len:] = 1
    with self.maybe_autocast():
        outputs = self.llama_model.generate(inputs_embeds=embs,
            attention_mask=attn_mask, max_new_tokens=max_new_tokens,
            num_beams=num_beams, length_penalty=length_penalty, temperature
            =temperature, do_sample=do_sample, min_length=min_length, top_p
            =top_p, repetition_penalty=repetition_penalty)
    answers = []
    for output_token in outputs:
        if output_token[0] == 0:
            output_token = output_token[1:]
        output_texts = self.llama_tokenizer.decode(output_token,
            skip_special_tokens=True)
        output_texts = output_texts.split('</s>')[0]
        output_texts = output_texts.replace('<s>', '')
        output_texts = output_texts.split('[/INST]')[-1].strip()
        answers.append(output_texts)
    return answers
