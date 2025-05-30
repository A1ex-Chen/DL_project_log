def load_textual_inversion(self, pretrained_model_name_or_path: Union[str,
    List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    token: Optional[Union[str, List[str]]]=None, **kwargs):
    """
        Load textual inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
        Automatic1111 formats are supported).

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike` or `List[str or os.PathLike]` or `Dict` or `List[Dict]`):
                Can be either one of the following or a list of them:

                    - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a
                      pretrained model hosted on the Hub.
                    - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual
                      inversion weights.
                    - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            token (`str` or `List[str]`, *optional*):
                Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a
                list, then `token` must also be a list of equal length.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used when:

                    - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
                      name such as `text_inv.bin`.
                    - The saved textual inversion file is in the Automatic1111 format.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        To load a textual inversion embedding vector in 🤗 Diffusers format:

        ```py
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

        pipe.load_textual_inversion("sd-concepts-library/cat-toy")

        prompt = "A <cat-toy> backpack"

        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("cat-backpack.png")
        ```

        To load a textual inversion embedding vector in Automatic1111 format, make sure to download the vector first
        (for example from [civitAI](https://civitai.com/models/3036?modelVersionId=9857)) and then load the vector
        locally:

        ```py
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

        pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")

        prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("character.png")
        ```

        """
    if not hasattr(self, 'tokenizer') or not isinstance(self.tokenizer,
        PreTrainedTokenizer):
        raise ValueError(
            f'{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling `{self.load_textual_inversion.__name__}`'
            )
    if not hasattr(self, 'text_encoder') or not isinstance(self.
        text_encoder, PreTrainedModel):
        raise ValueError(
            f'{self.__class__.__name__} requires `self.text_encoder` of type `PreTrainedModel` for calling `{self.load_textual_inversion.__name__}`'
            )
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop('use_auth_token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    weight_name = kwargs.pop('weight_name', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    if use_safetensors and not is_safetensors_available():
        raise ValueError(
            '`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors'
            )
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = is_safetensors_available()
        allow_pickle = True
    user_agent = {'file_type': 'text_inversion', 'framework': 'pytorch'}
    if not isinstance(pretrained_model_name_or_path, list):
        pretrained_model_name_or_paths = [pretrained_model_name_or_path]
    else:
        pretrained_model_name_or_paths = pretrained_model_name_or_path
    if isinstance(token, str):
        tokens = [token]
    elif token is None:
        tokens = [None] * len(pretrained_model_name_or_paths)
    else:
        tokens = token
    if len(pretrained_model_name_or_paths) != len(tokens):
        raise ValueError(
            f'You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)}Make sure both lists have the same length.'
            )
    valid_tokens = [t for t in tokens if t is not None]
    if len(set(valid_tokens)) < len(valid_tokens):
        raise ValueError(
            f'You have passed a list of tokens that contains duplicates: {tokens}'
            )
    token_ids_and_embeddings = []
    for pretrained_model_name_or_path, token in zip(
        pretrained_model_name_or_paths, tokens):
        if not isinstance(pretrained_model_name_or_path, dict):
            model_file = None
            if (use_safetensors and weight_name is None or weight_name is not
                None and weight_name.endswith('.safetensors')):
                try:
                    model_file = _get_model_file(pretrained_model_name_or_path,
                        weights_name=weight_name or
                        TEXT_INVERSION_NAME_SAFE, cache_dir=cache_dir,
                        force_download=force_download, resume_download=
                        resume_download, proxies=proxies, local_files_only=
                        local_files_only, use_auth_token=use_auth_token,
                        revision=revision, subfolder=subfolder, user_agent=
                        user_agent)
                    state_dict = safetensors.torch.load_file(model_file,
                        device='cpu')
                except Exception as e:
                    if not allow_pickle:
                        raise e
                    model_file = None
            if model_file is None:
                model_file = _get_model_file(pretrained_model_name_or_path,
                    weights_name=weight_name or TEXT_INVERSION_NAME,
                    cache_dir=cache_dir, force_download=force_download,
                    resume_download=resume_download, proxies=proxies,
                    local_files_only=local_files_only, use_auth_token=
                    use_auth_token, revision=revision, subfolder=subfolder,
                    user_agent=user_agent)
                state_dict = torch.load(model_file, map_location='cpu')
        else:
            state_dict = pretrained_model_name_or_path
        loaded_token = None
        if isinstance(state_dict, torch.Tensor):
            if token is None:
                raise ValueError(
                    'You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`.'
                    )
            embedding = state_dict
        elif len(state_dict) == 1:
            loaded_token, embedding = next(iter(state_dict.items()))
        elif 'string_to_param' in state_dict:
            loaded_token = state_dict['name']
            embedding = state_dict['string_to_param']['*']
        if token is not None and loaded_token != token:
            logger.info(
                f'The loaded token: {loaded_token} is overwritten by the passed token {token}.'
                )
        else:
            token = loaded_token
        embedding = embedding.to(dtype=self.text_encoder.dtype, device=self
            .text_encoder.device)
        vocab = self.tokenizer.get_vocab()
        if token in vocab:
            raise ValueError(
                f'Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder.'
                )
        elif f'{token}_1' in vocab:
            multi_vector_tokens = [token]
            i = 1
            while f'{token}_{i}' in self.tokenizer.added_tokens_encoder:
                multi_vector_tokens.append(f'{token}_{i}')
                i += 1
            raise ValueError(
                f'Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder.'
                )
        is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
        if is_multi_vector:
            tokens = [token] + [f'{token}_{i}' for i in range(1, embedding.
                shape[0])]
            embeddings = [e for e in embedding]
        else:
            tokens = [token]
            embeddings = [embedding[0]] if len(embedding.shape) > 1 else [
                embedding]
        self.tokenizer.add_tokens(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_and_embeddings += zip(token_ids, embeddings)
        logger.info(f'Loaded textual inversion embedding for {token}.')
    self.text_encoder.resize_token_embeddings(len(self.tokenizer))
    for token_id, embedding in token_ids_and_embeddings:
        self.text_encoder.get_input_embeddings().weight.data[token_id
            ] = embedding
