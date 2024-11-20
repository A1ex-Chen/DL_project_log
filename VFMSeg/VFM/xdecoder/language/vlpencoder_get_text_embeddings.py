def get_text_embeddings(self, class_names, name='default', is_eval=False,
    add_bgd=False, prompt=True, norm=True):
    if not is_eval:
        if prompt:
            arbitary_concepts = [prompt_engineering(class_names[label].
                replace('-other', '').replace('-merged', '').replace(
                '-stuff', ''), topk=10000, suffix='.') for label in range(
                len(class_names))]
            if add_bgd:
                arbitary_concepts.append('A background in coco.')
        else:
            arbitary_concepts = class_names
        input_ids = []
        attention_masks = []
        for txt in arbitary_concepts:
            tokens = self.tokenizer(txt, padding='max_length', truncation=
                True, max_length=self.max_token_num, return_tensors='pt')
            tokens['input_ids'].squeeze_()
            tokens['attention_mask'].squeeze_()
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])
        arbitary_tokens = torch.stack(input_ids)
        arbitary_attention_masks = torch.stack(attention_masks)
        text_emb = self.forward_language((arbitary_tokens.cuda(),
            arbitary_attention_masks.cuda()), norm=norm)
        setattr(self, '{}_text_embeddings'.format(name), text_emb)
    else:
        with torch.no_grad():

            def extract_mean_emb(txts):
                tokens = self.tokenizer(txts, padding='max_length',
                    truncation=True, max_length=self.max_token_num,
                    return_tensors='pt')
                clss_embedding = self.forward_language((tokens['input_ids']
                    .cuda(), tokens['attention_mask'].cuda()), norm=norm)
                clss_embedding = clss_embedding.mean(dim=0)
                clss_embedding /= clss_embedding.norm()
                return clss_embedding
            templates = get_prompt_templates()
            clss_embeddings = []
            if prompt:
                for clss in class_names:
                    txts = [template.format(clss.replace('-other', '').
                        replace('-merged', '').replace('-stuff', '')) for
                        template in templates]
                    clss_embeddings.append(extract_mean_emb(txts))
            else:
                clss_embeddings.append(extract_mean_emb(class_names))
            if add_bgd:
                txts = ['A background in coco.']
                clss_embeddings.append(extract_mean_emb(txts))
            text_emb = torch.stack(clss_embeddings, dim=0)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
