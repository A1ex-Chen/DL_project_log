def pipeline_coreml(model, im, file, names, y, prefix=colorstr(
    'CoreML Pipeline:')):
    import coremltools as ct
    from PIL import Image
    print(f'{prefix} starting pipeline with coremltools {ct.__version__}...')
    batch_size, ch, h, w = list(im.shape)
    t = time.time()
    spec = model.get_spec()
    out0, out1 = iter(spec.description.output)
    if platform.system() == 'Darwin':
        img = Image.new('RGB', (w, h))
        out = model.predict({'image': img})
        out0_shape, out1_shape = out[out0.name].shape, out[out1.name].shape
    else:
        s = tuple(y[0].shape)
        out0_shape, out1_shape = (s[1], s[2] - 5), (s[1], 4)
    nx, ny = spec.description.input[0
        ].type.imageType.width, spec.description.input[0].type.imageType.height
    na, nc = out0_shape
    assert len(names) == nc, f'{len(names)} names found for nc={nc}'
    out0.type.multiArrayType.shape[:] = out0_shape
    out1.type.multiArrayType.shape[:] = out1_shape
    print(spec.description)
    model = ct.models.MLModel(spec)
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)
    nms_spec.description.output[0].name = 'confidence'
    nms_spec.description.output[1].name = 'coordinates'
    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]
    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name
    nms.coordinatesInputFeatureName = out1.name
    nms.confidenceOutputFeatureName = 'confidence'
    nms.coordinatesOutputFeatureName = 'coordinates'
    nms.iouThresholdInputFeatureName = 'iouThreshold'
    nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.pickTop.perClass = True
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)
    pipeline = ct.models.pipeline.Pipeline(input_features=[('image', ct.
        models.datatypes.Array(3, ny, nx)), ('iouThreshold', ct.models.
        datatypes.Double()), ('confidenceThreshold', ct.models.datatypes.
        Double())], output_features=['confidence', 'coordinates'])
    pipeline.add_model(model)
    pipeline.add_model(nms_model)
    pipeline.spec.description.input[0].ParseFromString(model._spec.
        description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.
        description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.
        description.output[1].SerializeToString())
    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.versionString = (
        'https://github.com/ultralytics/yolov5')
    pipeline.spec.description.metadata.shortDescription = (
        'https://github.com/ultralytics/yolov5')
    pipeline.spec.description.metadata.author = 'glenn.jocher@ultralytics.com'
    pipeline.spec.description.metadata.license = (
        'https://github.com/ultralytics/yolov5/blob/master/LICENSE')
    pipeline.spec.description.metadata.userDefined.update({'classes': ','.
        join(names.values()), 'iou_threshold': str(nms.iouThreshold),
        'confidence_threshold': str(nms.confidenceThreshold)})
    f = file.with_suffix('.mlmodel')
    model = ct.models.MLModel(pipeline.spec)
    model.input_description['image'] = 'Input image'
    model.input_description['iouThreshold'
        ] = f'(optional) IOU Threshold override (default: {nms.iouThreshold})'
    model.input_description['confidenceThreshold'] = (
        f'(optional) Confidence Threshold override (default: {nms.confidenceThreshold})'
        )
    model.output_description['confidence'
        ] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description['coordinates'
        ] = 'Boxes × [x, y, width, height] (relative to image size)'
    model.save(f)
    print(
        f'{prefix} pipeline success ({time.time() - t:.2f}s), saved as {f} ({file_size(f):.1f} MB)'
        )
