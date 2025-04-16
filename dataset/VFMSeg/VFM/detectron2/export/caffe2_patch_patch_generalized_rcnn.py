def patch_generalized_rcnn(model):
    ccc = Caffe2CompatibleConverter
    model = patch(model, rpn.RPN, ccc(Caffe2RPN))
    model = patch(model, poolers.ROIPooler, ccc(Caffe2ROIPooler))
    return model
