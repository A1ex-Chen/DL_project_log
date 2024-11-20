def pre_caption(self, caption):
    caption = re.sub('([.!\\"()*#:;~])', ' ', caption.lower())
    caption = re.sub('\\s{2,}', ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words) > self.max_words:
        caption = ' '.join(caption_words[:self.max_words])
    return caption
