def batch_generate(self, image_list, question_list, max_new_tokens, **kwargs):
    """
            process a batch of images and questions, and then do_generate
        """
    answers = []
    for imgs, prompts in zip(image_list, question_list):
        if isinstance(imgs, str):
            imgs = [imgs]
        if len(imgs) > 1:
            prompts += (
                ' The order in which I upload images is the order of the images.'
                )
        print(imgs, prompts)
        msg = create_message(imgs, prompts)
        try:
            response = self.client.chat.completions.create(model=self.
                gpt_name, messages=msg, max_tokens=max_new_tokens)
        except Exception as e:
            error_msg = str(e).split('message')[1][4:].split("'")[0]
            answers.append('##Error##:' + error_msg)
            continue
        answers.append(response.choices[0].message.content)
        print(answers[-1])
    """
            Direct generate answers with single image and questions, max_len(answer) = max_new_tokens
        """
    return answers
