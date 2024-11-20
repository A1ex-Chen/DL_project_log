def test_from_pipe_consistent(self):
    for model_name, model_repo in PRETRAINED_MODEL_REPO_MAPPING.items():
        if model_name in ['kandinsky', 'kandinsky22']:
            auto_pipes = [AutoPipelineForText2Image, AutoPipelineForImage2Image
                ]
        else:
            auto_pipes = [AutoPipelineForText2Image,
                AutoPipelineForImage2Image, AutoPipelineForInpainting]
        for pipe_from_class in auto_pipes:
            pipe_from = pipe_from_class.from_pretrained(model_repo, variant
                ='fp16', torch_dtype=torch.float16)
            pipe_from_config = dict(pipe_from.config)
            for pipe_to_class in auto_pipes:
                pipe_to = pipe_to_class.from_pipe(pipe_from)
                self.assertEqual(dict(pipe_to.config), pipe_from_config)
            del pipe_from, pipe_to
            gc.collect()
