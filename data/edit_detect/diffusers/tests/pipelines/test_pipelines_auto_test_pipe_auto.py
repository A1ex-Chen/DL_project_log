def test_pipe_auto(self):
    for model_name, model_repo in PRETRAINED_MODEL_REPO_MAPPING.items():
        pipe_txt2img = AutoPipelineForText2Image.from_pretrained(model_repo,
            variant='fp16', torch_dtype=torch.float16)
        self.assertIsInstance(pipe_txt2img,
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])
        pipe_to = AutoPipelineForText2Image.from_pipe(pipe_txt2img)
        self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[
            model_name])
        pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_txt2img)
        self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[
            model_name])
        if 'kandinsky' not in model_name:
            pipe_to = AutoPipelineForInpainting.from_pipe(pipe_txt2img)
            self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[
                model_name])
        del pipe_txt2img, pipe_to
        gc.collect()
        pipe_img2img = AutoPipelineForImage2Image.from_pretrained(model_repo,
            variant='fp16', torch_dtype=torch.float16)
        self.assertIsInstance(pipe_img2img,
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])
        pipe_to = AutoPipelineForText2Image.from_pipe(pipe_img2img)
        self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[
            model_name])
        pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_img2img)
        self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[
            model_name])
        if 'kandinsky' not in model_name:
            pipe_to = AutoPipelineForInpainting.from_pipe(pipe_img2img)
            self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[
                model_name])
        del pipe_img2img, pipe_to
        gc.collect()
        if 'kandinsky' not in model_name:
            pipe_inpaint = AutoPipelineForInpainting.from_pretrained(model_repo
                , variant='fp16', torch_dtype=torch.float16)
            self.assertIsInstance(pipe_inpaint,
                AUTO_INPAINT_PIPELINES_MAPPING[model_name])
            pipe_to = AutoPipelineForText2Image.from_pipe(pipe_inpaint)
            self.assertIsInstance(pipe_to,
                AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])
            pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_inpaint)
            self.assertIsInstance(pipe_to,
                AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])
            pipe_to = AutoPipelineForInpainting.from_pipe(pipe_inpaint)
            self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[
                model_name])
            del pipe_inpaint, pipe_to
            gc.collect()
