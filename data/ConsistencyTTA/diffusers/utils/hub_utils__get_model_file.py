def _get_model_file(pretrained_model_name_or_path, *, weights_name,
    subfolder, cache_dir, force_download, proxies, resume_download,
    local_files_only, use_auth_token, user_agent, revision, commit_hash=None):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path,
            weights_name)):
            model_file = os.path.join(pretrained_model_name_or_path,
                weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(os.path.join(
            pretrained_model_name_or_path, subfolder, weights_name)):
            model_file = os.path.join(pretrained_model_name_or_path,
                subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f'Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.'
                )
    else:
        if revision in DEPRECATED_REVISION_ARGS and (weights_name ==
            WEIGHTS_NAME or weights_name == SAFETENSORS_WEIGHTS_NAME
            ) and version.parse(version.parse(__version__).base_version
            ) >= version.parse('0.17.0'):
            try:
                model_file = hf_hub_download(pretrained_model_name_or_path,
                    filename=_add_variant(weights_name, revision),
                    cache_dir=cache_dir, force_download=force_download,
                    proxies=proxies, resume_download=resume_download,
                    local_files_only=local_files_only, use_auth_token=
                    use_auth_token, user_agent=user_agent, subfolder=
                    subfolder, revision=revision or commit_hash)
                warnings.warn(
                    f"Loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` is deprecated. Loading instead from `revision='main'` with `variant={revision}`. Loading model variants via `revision='{revision}'` will be removed in diffusers v1. Please use `variant='{revision}'` instead."
                    , FutureWarning)
                return model_file
            except:
                warnings.warn(
                    f"""You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have a {_add_variant(weights_name, revision)} file in the 'main' branch of {pretrained_model_name_or_path}. 
 The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {_add_variant(weights_name, revision)}' so that the correct variant file can be added."""
                    , FutureWarning)
        try:
            model_file = hf_hub_download(pretrained_model_name_or_path,
                filename=weights_name, cache_dir=cache_dir, force_download=
                force_download, proxies=proxies, resume_download=
                resume_download, local_files_only=local_files_only,
                use_auth_token=use_auth_token, user_agent=user_agent,
                subfolder=subfolder, revision=revision or commit_hash)
            return model_file
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
                f'{pretrained_model_name_or_path} does not appear to have a file named {weights_name}.'
                )
        except HTTPError as err:
            raise EnvironmentError(
                f"""There was a specific connection error when trying to load {pretrained_model_name_or_path}:
{err}"""
                )
        except ValueError:
            raise EnvironmentError(
                f"""We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a directory containing a file named {weights_name} or 
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."""
                )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {weights_name}"
                )
