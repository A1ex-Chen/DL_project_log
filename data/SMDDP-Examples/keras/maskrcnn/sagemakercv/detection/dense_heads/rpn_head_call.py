def call(self, inputs, img_info, training=True, *args, **kwargs):
    scores_outputs = dict()
    box_outputs = dict()
    for level in range(self.anchor_generator.min_level, self.
        anchor_generator.max_level + 1):
        net = self.rpn_conv(inputs[level])
        scores_outputs[level] = self.conv_cls(net)
        box_outputs[level] = self.conv_reg(net)
    proposals = self.get_bboxes(scores_outputs, box_outputs, img_info, self
        .anchor_generator, training=training)
    return scores_outputs, box_outputs, proposals
