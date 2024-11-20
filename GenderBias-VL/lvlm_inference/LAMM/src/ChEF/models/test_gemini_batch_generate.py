def batch_generate(self, image_list, question_list, max_new_tokens, **kwargs):
    """
            process a batch of images and questions, and then do_generate
        """
    answers = []
    generation_config = genai.types.GenerationConfig(candidate_count=1,
        max_output_tokens=max_new_tokens, temperature=1.0)
    for imgs, prompts in zip(image_list, question_list):
        print(imgs, prompts)
        if isinstance(imgs, str):
            imgs = [imgs]
        img_list = [Image(img) for img in imgs]
        try_time = 0
        response = None
        while True:
            try:
                if self.safety_block_none:
                    response = self.model.generate_content(contents=[
                        prompts] + img_list, safety_settings=
                        safety_settings, generation_config=generation_config)
                else:
                    response = self.model.generate_content(contents=[
                        prompts] + img_list, generation_config=
                        generation_config)
                response.resolve()
                answers.append(str(response.text).strip())
                print(answers[-1])
                try_time = 0
                break
            except Exception as e:
                if try_time >= 5:
                    try:
                        answers.append('##ERROR## ' + str(response.
                            prompt_feedback).strip())
                    except:
                        answers.append('##ERROR##')
                    break
                try_time += 1
                continue
    """
            Direct generate answers with single image and questions, max_len(answer) = max_new_tokens
        """
    return answers
