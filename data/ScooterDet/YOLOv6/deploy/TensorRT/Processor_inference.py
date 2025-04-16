def inference(self, inputs):
    self.binding_addrs[self.input_names[0]] = int(inputs.data_ptr())
    self.context.execute_v2(list(self.binding_addrs.values()))
    if self.is_end2end:
        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data
        output = torch.cat((boxes, scores[:, :, None], classes[:, :, None]),
            axis=-1)
    else:
        output = self.bindings[self.output_names[0]].data
    return output
