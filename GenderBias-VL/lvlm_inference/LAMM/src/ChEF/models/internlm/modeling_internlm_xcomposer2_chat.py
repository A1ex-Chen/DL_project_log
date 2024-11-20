@torch.no_grad()
def chat(self, tokenizer, query: str, image: torch.Tensor=None, history:
    List[Tuple[str, str]]=[], streamer: Optional[BaseStreamer]=None,
    max_new_tokens: int=1024, do_sample: bool=True, temperature: float=1.0,
    top_p: float=0.8, repetition_penalty: float=1.005, meta_instruction:
    str=
    """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image."""
    , **kwargs):
    if image is None:
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
    else:
        image = self.encode_img(image)
        inputs, im_mask = self.interleav_wrap_chat(tokenizer, query, image,
            history, meta_instruction)
    inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.
        is_tensor(v)}
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids
        (['[UNUSED_TOKEN_145]'])[0]]
    outputs = self.generate(**inputs, streamer=streamer, max_new_tokens=
        max_new_tokens, do_sample=do_sample, temperature=temperature, top_p
        =top_p, eos_token_id=eos_token_id, repetition_penalty=
        repetition_penalty, im_mask=im_mask, **kwargs)
    if image is None:
        outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]):]
    else:
        outputs = outputs[0].cpu().tolist()
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    response = response.split('[UNUSED_TOKEN_145]')[0]
    history = history + [(query, response)]
    return response, history
