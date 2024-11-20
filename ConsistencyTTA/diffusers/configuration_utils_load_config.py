@classmethod
def load_config(cls, pretrained_model_name_or_path: Union[str, os.PathLike],
    return_unused_kwargs=False, return_commit_hash=False, **kwargs) ->Tuple[
    Dict[str, Any], Dict[str, Any]]:
    """
        Load a model or scheduler configuration.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing model weights saved with
                      [`~ConfigMixin.save_config`].

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
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
            return_unused_kwargs (`bool`, *optional*, defaults to `False):
                Whether unused keyword arguments of the config are returned.
            return_commit_hash (`bool`, *optional*, defaults to `False):
                Whether the `commit_hash` of the loaded configuration are returned.

        Returns:
            `dict`:
                A dictionary of all the parameters stored in a JSON configuration file.

        """
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    use_auth_token = kwargs.pop('use_auth_token', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    _ = kwargs.pop('mirror', None)
    subfolder = kwargs.pop('subfolder', None)
    user_agent = kwargs.pop('user_agent', {})
    user_agent = {**user_agent, 'file_type': 'config'}
    user_agent = http_user_agent(user_agent)
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if cls.config_name is None:
        raise ValueError(
            '`self.config_name` is not defined. Note that one should not load a config from `ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`'
            )
    if os.path.isfile(pretrained_model_name_or_path):
        config_file = pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.
            config_name)):
            config_file = os.path.join(pretrained_model_name_or_path, cls.
                config_name)
        elif subfolder is not None and os.path.isfile(os.path.join(
            pretrained_model_name_or_path, subfolder, cls.config_name)):
            config_file = os.path.join(pretrained_model_name_or_path,
                subfolder, cls.config_name)
        else:
            raise EnvironmentError(
                f'Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}.'
                )
    else:
        try:
            config_file = hf_hub_download(pretrained_model_name_or_path,
                filename=cls.config_name, cache_dir=cache_dir,
                force_download=force_download, proxies=proxies,
                resume_download=resume_download, local_files_only=
                local_files_only, use_auth_token=use_auth_token, user_agent
                =user_agent, subfolder=subfolder, revision=revision)
        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"""{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo with `use_auth_token` or log in with `huggingface-cli login`."""
                )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
        except EntryNotFoundError:
            raise EnvironmentError(
                f'{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}.'
                )
        except HTTPError as err:
            raise EnvironmentError(
                f"""There was a specific connection error when trying to load {pretrained_model_name_or_path}:
{err}"""
                )
        except ValueError:
            raise EnvironmentError(
                f"""We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a directory containing a {cls.config_name} file.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."""
                )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a {cls.config_name} file"
                )
    try:
        config_dict = cls._dict_from_json_file(config_file)
        commit_hash = extract_commit_hash(config_file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EnvironmentError(
            f"It looks like the config file at '{config_file}' is not a valid JSON file."
            )
    if not (return_unused_kwargs or return_commit_hash):
        return config_dict
    outputs = config_dict,
    if return_unused_kwargs:
        outputs += kwargs,
    if return_commit_hash:
        outputs += commit_hash,
    return outputs
