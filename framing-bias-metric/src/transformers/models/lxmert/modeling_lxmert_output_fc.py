def output_fc(self, lang_input, visual_input):
    lang_inter_output = self.lang_inter(lang_input)
    visual_inter_output = self.visn_inter(visual_input)
    lang_output = self.lang_output(lang_inter_output, lang_input)
    visual_output = self.visn_output(visual_inter_output, visual_input)
    return lang_output, visual_output
