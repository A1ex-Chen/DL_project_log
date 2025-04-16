def push_to_hub(self, repo_id: str, commit_message: Optional[str]=None,
    private: Optional[bool]=None, token: Optional[str]=None, create_pr:
    bool=False, safe_serialization: bool=True, variant: Optional[str]=None
    ) ->str:
    """
        Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your model, scheduler, or pipeline files to. It should
                contain your organization name when pushing to an organization. `repo_id` can also be a path to a local
                directory.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. The token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights to the `safetensors` format.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.

        Examples:

        ```python
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")

        # Push the `unet` to your namespace with the name "my-finetuned-unet".
        unet.push_to_hub("my-finetuned-unet")

        # Push the `unet` to an organization with the name "my-finetuned-unet".
        unet.push_to_hub("your-org/my-finetuned-unet")
        ```
        """
    repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True
        ).repo_id
    model_card = load_or_create_model_card(repo_id, token=token)
    model_card = populate_model_card(model_card)
    save_kwargs = {'safe_serialization': safe_serialization}
    if 'Scheduler' not in self.__class__.__name__:
        save_kwargs.update({'variant': variant})
    with tempfile.TemporaryDirectory() as tmpdir:
        self.save_pretrained(tmpdir, **save_kwargs)
        model_card.save(os.path.join(tmpdir, 'README.md'))
        return self._upload_folder(tmpdir, repo_id, token=token,
            commit_message=commit_message, create_pr=create_pr)
