def _add_tflite_metadata(self, file):
    """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata."""
    import flatbuffers
    if ARM64:
        from tflite_support import metadata
        from tflite_support import metadata_schema_py_generated as schema
    else:
        from tensorflow_lite_support.metadata import metadata_schema_py_generated as schema
        from tensorflow_lite_support.metadata.python import metadata
    model_meta = schema.ModelMetadataT()
    model_meta.name = self.metadata['description']
    model_meta.version = self.metadata['version']
    model_meta.author = self.metadata['author']
    model_meta.license = self.metadata['license']
    tmp_file = Path(file).parent / 'temp_meta.txt'
    with open(tmp_file, 'w') as f:
        f.write(str(self.metadata))
    label_file = schema.AssociatedFileT()
    label_file.name = tmp_file.name
    label_file.type = schema.AssociatedFileType.TENSOR_AXIS_LABELS
    input_meta = schema.TensorMetadataT()
    input_meta.name = 'image'
    input_meta.description = 'Input image to be detected.'
    input_meta.content = schema.ContentT()
    input_meta.content.contentProperties = schema.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = schema.ColorSpaceType.RGB
    input_meta.content.contentPropertiesType = (schema.ContentProperties.
        ImageProperties)
    output1 = schema.TensorMetadataT()
    output1.name = 'output'
    output1.description = (
        'Coordinates of detected objects, class labels, and confidence score')
    output1.associatedFiles = [label_file]
    if self.model.task == 'segment':
        output2 = schema.TensorMetadataT()
        output2.name = 'output'
        output2.description = 'Mask protos'
        output2.associatedFiles = [label_file]
    subgraph = schema.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output1, output2
        ] if self.model.task == 'segment' else [output1]
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(model_meta.Pack(b), metadata.MetadataPopulator.
        METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()
    populator = metadata.MetadataPopulator.with_model_file(str(file))
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([str(tmp_file)])
    populator.populate()
    tmp_file.unlink()
