def add_tflite_metadata(file, metadata, num_outputs):
    with contextlib.suppress(ImportError):
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb
        tmp_file = Path('/tmp/meta.txt')
        with open(tmp_file, 'w') as meta_f:
            meta_f.write(str(metadata))
        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()
            ] * num_outputs
        model_meta.subgraphMetadata = [subgraph]
        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.
            METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()
        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()
