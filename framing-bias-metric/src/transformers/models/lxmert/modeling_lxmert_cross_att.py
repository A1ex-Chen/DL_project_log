def cross_att(self, lang_input, lang_attention_mask, visual_input,
    visual_attention_mask, output_x_attentions=False):
    lang_att_output = self.visual_attention(lang_input, visual_input,
        ctx_att_mask=visual_attention_mask, output_attentions=
        output_x_attentions)
    visual_att_output = self.visual_attention(visual_input, lang_input,
        ctx_att_mask=lang_attention_mask, output_attentions=False)
    return lang_att_output, visual_att_output
