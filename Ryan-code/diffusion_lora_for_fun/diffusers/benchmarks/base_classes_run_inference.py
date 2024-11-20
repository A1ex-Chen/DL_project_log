def run_inference(self, pipe, args):
    _ = pipe(prompt=PROMPT, image=self.image, num_inference_steps=args.
        num_inference_steps, num_images_per_prompt=args.batch_size)
