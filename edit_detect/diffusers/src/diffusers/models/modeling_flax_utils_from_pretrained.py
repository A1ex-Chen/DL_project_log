@classmethod
@validate_hf_hub_args
def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.
    PathLike], dtype: jnp.dtype=jnp.float32, *model_args, **kwargs):
    """
        Instantiate a pretrained Flax model from a pretrained model configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* (for example `runwayml/stable-diffusion-v1-5`) of a pretrained model
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      using [`~FlaxModelMixin.save_pretrained`].
            dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
                The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
                `jax.numpy.bfloat16` (on TPUs).

                This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
                specified, all the computation will be performed with the given `dtype`.

                <Tip>

                This only specifies the dtype of the *computation* and does not influence the dtype of model
                parameters.

                If you wish to change the dtype of the model parameters, see [`~FlaxModelMixin.to_fp16`] and
                [`~FlaxModelMixin.to_bf16`].

                </Tip>

            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments are passed to the underlying model's `__init__` method.
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
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it is loaded) and initiate the model (for
                example, `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `kwargs` are directly passed to the underlying
                      model's `__init__` method (we assume all relevant updates to the configuration have already been
                      done).
                    - If a configuration is not provided, `kwargs` are first passed to the configuration class
                      initialization function [`~ConfigMixin.from_config`]. Each key of the `kwargs` that corresponds
                      to a configuration attribute is used to override said attribute with the supplied `kwargs` value.
                      Remaining keys that do not correspond to any configuration attribute are passed to the underlying
                      model's `__init__` function.

        Examples:

        ```python
        >>> from diffusers import FlaxUNet2DConditionModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model, params = FlaxUNet2DConditionModel.from_pretrained("./test/saved_model/")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
    config = kwargs.pop('config', None)
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    from_pt = kwargs.pop('from_pt', False)
    resume_download = kwargs.pop('resume_download', None)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    user_agent = {'diffusers': __version__, 'file_type': 'model',
        'framework': 'flax'}
    if config is None:
        config, unused_kwargs = cls.load_config(pretrained_model_name_or_path,
            cache_dir=cache_dir, return_unused_kwargs=True, force_download=
            force_download, resume_download=resume_download, proxies=
            proxies, local_files_only=local_files_only, token=token,
            revision=revision, subfolder=subfolder, **kwargs)
    model, model_kwargs = cls.from_config(config, dtype=dtype,
        return_unused_kwargs=True, **unused_kwargs)
    pretrained_path_with_subfolder = (pretrained_model_name_or_path if 
        subfolder is None else os.path.join(pretrained_model_name_or_path,
        subfolder))
    if os.path.isdir(pretrained_path_with_subfolder):
        if from_pt:
            if not os.path.isfile(os.path.join(
                pretrained_path_with_subfolder, WEIGHTS_NAME)):
                raise EnvironmentError(
                    f'Error no file named {WEIGHTS_NAME} found in directory {pretrained_path_with_subfolder} '
                    )
            model_file = os.path.join(pretrained_path_with_subfolder,
                WEIGHTS_NAME)
        elif os.path.isfile(os.path.join(pretrained_path_with_subfolder,
            FLAX_WEIGHTS_NAME)):
            model_file = os.path.join(pretrained_path_with_subfolder,
                FLAX_WEIGHTS_NAME)
        elif os.path.isfile(os.path.join(pretrained_path_with_subfolder,
            WEIGHTS_NAME)):
            raise EnvironmentError(
                f'{WEIGHTS_NAME} file found in directory {pretrained_path_with_subfolder}. Please load the model using `from_pt=True`.'
                )
        else:
            raise EnvironmentError(
                f'Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory {pretrained_path_with_subfolder}.'
                )
    else:
        try:
            model_file = hf_hub_download(pretrained_model_name_or_path,
                filename=FLAX_WEIGHTS_NAME if not from_pt else WEIGHTS_NAME,
                cache_dir=cache_dir, force_download=force_download, proxies
                =proxies, resume_download=resume_download, local_files_only
                =local_files_only, token=token, user_agent=user_agent,
                subfolder=subfolder, revision=revision)
        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"""{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo with `token` or log in with `huggingface-cli login`."""
                )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
        except EntryNotFoundError:
            raise EnvironmentError(
                f'{pretrained_model_name_or_path} does not appear to have a file named {FLAX_WEIGHTS_NAME}.'
                )
        except HTTPError as err:
            raise EnvironmentError(
                f"""There was a specific connection error when trying to load {pretrained_model_name_or_path}:
{err}"""
                )
        except ValueError:
            raise EnvironmentError(
                f"""We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."""
                )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                )
    if from_pt:
        if is_torch_available():
            from .modeling_utils import load_state_dict
        else:
            raise EnvironmentError(
                "Can't load the model in PyTorch format because PyTorch is not installed. Please, install PyTorch or use native Flax weights."
                )
        pytorch_model_file = load_state_dict(model_file)
        state = convert_pytorch_state_dict_to_flax(pytorch_model_file, model)
    else:
        try:
            with open(model_file, 'rb') as state_f:
                state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                with open(model_file) as f:
                    if f.read().startswith('version'):
                        raise OSError(
                            'You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.'
                            )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(
                    f'Unable to convert {model_file} to Flax deserializable object. '
                    )
    state = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.
        local_devices(backend='cpu')[0]), state)
    state = flatten_dict(state)
    params_shape_tree = jax.eval_shape(model.init_weights, rng=jax.random.
        PRNGKey(0))
    required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())
    shape_state = flatten_dict(unfreeze(params_shape_tree))
    missing_keys = required_params - set(state.keys())
    unexpected_keys = set(state.keys()) - required_params
    if missing_keys:
        logger.warning(
            f'The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. Make sure to call model.init_weights to initialize the missing weights.'
            )
        cls._missing_keys = missing_keys
    for key in state.keys():
        if key in shape_state and state[key].shape != shape_state[key].shape:
            raise ValueError(
                f'Trying to load the pretrained weight for {key} failed: checkpoint has shape {state[key].shape} which is incompatible with the model shape {shape_state[key].shape}. '
                )
    for unexpected_key in unexpected_keys:
        del state[unexpected_key]
    if len(unexpected_keys) > 0:
        logger.warning(
            f"""Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}
- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture."""
            )
    else:
        logger.info(
            f"""All model checkpoint weights were used when initializing {model.__class__.__name__}.
"""
            )
    if len(missing_keys) > 0:
        logger.warning(
            f"""Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    else:
        logger.info(
            f"""All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.
If your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training."""
            )
    return model, unflatten_dict(state)
