def self_att(self, lang_input, lang_attention_mask, visn_input,
    visn_attention_mask, training=False):
    output_attentions = False
    lang_att_output = self.lang_self_att(lang_input, lang_attention_mask,
        output_attentions, training=training)
    visn_att_output = self.visn_self_att(visn_input, visn_attention_mask,
        output_attentions, training=training)
    return lang_att_output[0], visn_att_output[0]
