@validate_hf_hub_args
def load_textual_inversion(self, pretrained_model_name_or_path: Union[str,
    List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    token: Optional[Union[str, List[str]]]=None, tokenizer: Optional[
    'PreTrainedTokenizer']=None, text_encoder: Optional['PreTrainedModel']=
    None, **kwargs):
    """
        Load Textual Inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
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
            text_encoder ([`~transformers.CLIPTextModel`], *optional*):
                Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
                If not specified, function will take self.tokenizer.
            tokenizer ([`~transformers.CLIPTokenizer`], *optional*):
                A `CLIPTokenizer` to tokenize text. If not specified, function will take self.tokenizer.
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
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible. Will be removed in v1
                of Diffusers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
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

        To load a Textual Inversion embedding vector in 🤗 Diffusers format:

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

        To load a Textual Inversion embedding vector in Automatic1111 format, make sure to download the vector first
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
    tokenizer = tokenizer or getattr(self, 'tokenizer', None)
    text_encoder = text_encoder or getattr(self, 'text_encoder', None)
    pretrained_model_name_or_paths = [pretrained_model_name_or_path
        ] if not isinstance(pretrained_model_name_or_path, list
        ) else pretrained_model_name_or_path
    tokens = [token] if not isinstance(token, list) else token
    if tokens[0] is None:
        tokens = tokens * len(pretrained_model_name_or_paths)
    self._check_text_inv_inputs(tokenizer, text_encoder,
        pretrained_model_name_or_paths, tokens)
    state_dicts = load_textual_inversion_state_dicts(
        pretrained_model_name_or_paths, **kwargs)
    if len(tokens) > 1 and len(state_dicts) == 1:
        if isinstance(state_dicts[0], torch.Tensor):
            state_dicts = list(state_dicts[0])
            if len(tokens) != len(state_dicts):
                raise ValueError(
                    f'You have passed a state_dict contains {len(state_dicts)} embeddings, and list of tokens of length {len(tokens)} Make sure both have the same length.'
                    )
    tokens, embeddings = self._retrieve_tokens_and_embeddings(tokens,
        state_dicts, tokenizer)
    tokens, embeddings = self._extend_tokens_and_embeddings(tokens,
        embeddings, tokenizer)
    expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
    if any(expected_emb_dim != emb.shape[-1] for emb in embeddings):
        raise ValueError(
            'Loaded embeddings are of incorrect shape. Expected each textual inversion embedding to be of shape {input_embeddings.shape[-1]}, but are {embeddings.shape[-1]} '
            )
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    if self.hf_device_map is None:
        for _, component in self.components.items():
            if isinstance(component, nn.Module):
                if hasattr(component, '_hf_hook'):
                    is_model_cpu_offload = isinstance(getattr(component,
                        '_hf_hook'), CpuOffload)
                    is_sequential_cpu_offload = isinstance(getattr(
                        component, '_hf_hook'), AlignDevicesHook) or hasattr(
                        component._hf_hook, 'hooks') and isinstance(component
                        ._hf_hook.hooks[0], AlignDevicesHook)
                    logger.info(
                        'Accelerate hooks detected. Since you have called `load_textual_inversion()`, the previous hooks will be first removed. Then the textual inversion parameters will be loaded and the hooks will be applied again.'
                        )
                    remove_hook_from_module(component, recurse=
                        is_sequential_cpu_offload)
    device = text_encoder.device
    dtype = text_encoder.dtype
    text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))
    input_embeddings = text_encoder.get_input_embeddings().weight
    for token, embedding in zip(tokens, embeddings):
        tokenizer.add_tokens(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        input_embeddings.data[token_id] = embedding
        logger.info(f'Loaded textual inversion embedding for {token}.')
    input_embeddings.to(dtype=dtype, device=device)
    if is_model_cpu_offload:
        self.enable_model_cpu_offload()
    elif is_sequential_cpu_offload:
        self.enable_sequential_cpu_offload()
