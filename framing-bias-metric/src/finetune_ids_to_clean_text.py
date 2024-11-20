def ids_to_clean_text(self, generated_ids: List[int]):
    gen_text = self.tokenizer.batch_decode(generated_ids,
        skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return lmap(str.strip, gen_text)
