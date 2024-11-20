def output_fc(self, lang_input, visn_input, training=False):
    lang_inter_output = self.lang_inter(lang_input)
    visn_inter_output = self.visn_inter(visn_input)
    lang_output = self.lang_output(lang_inter_output, lang_input, training)
    visn_output = self.visn_output(visn_inter_output, visn_input, training)
    return lang_output, visn_output
