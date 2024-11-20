def cross_att(self, lang_input, lang_attention_mask, visn_input,
    visn_attention_mask, output_attentions, training=False):
    lang_attention_lang_input = tf.identity(lang_input)
    visn_attention_lang_input = tf.identity(lang_input)
    lang_attention_visn_input = tf.identity(visn_input)
    visn_attention_visn_input = tf.identity(visn_input)
    lang_att_output = self.visual_attention(lang_attention_lang_input,
        lang_attention_visn_input, visn_attention_mask, output_attentions=
        output_attentions, training=training)
    visn_att_output = self.visual_attention(visn_attention_visn_input,
        visn_attention_lang_input, lang_attention_mask, output_attentions=
        output_attentions, training=training)
    return lang_att_output, visn_att_output
