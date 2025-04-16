def run(self):
    if version.parse(huggingface_hub.__version__) < version.parse('0.9.0'):
        raise ImportError(
            'The huggingface_hub version must be >= 0.9.0 to use this command. Please update your huggingface_hub installation.'
            )
    else:
        from huggingface_hub import create_commit
        from huggingface_hub._commit_api import CommitOperationAdd
    model_index = hf_hub_download(repo_id=self.ckpt_id, filename=
        'model_index.json')
    with open(model_index, 'r') as f:
        pipeline_class_name = json.load(f)['_class_name']
    pipeline_class = getattr(import_module('diffusers'), pipeline_class_name)
    self.logger.info(f'Pipeline class imported: {pipeline_class_name}.')
    pipeline = pipeline_class.from_pretrained(self.ckpt_id, torch_dtype=
        torch.float16 if self.fp16 else torch.float32)
    pipeline.save_pretrained(self.local_ckpt_dir, safe_serialization=True if
        self.use_safetensors else False, variant='fp16' if self.fp16 else None)
    self.logger.info(f'Pipeline locally saved to {self.local_ckpt_dir}.')
    if self.fp16:
        modified_paths = glob.glob(f'{self.local_ckpt_dir}/*/*.fp16.*')
    elif self.use_safetensors:
        modified_paths = glob.glob(f'{self.local_ckpt_dir}/*/*.safetensors')
    commit_message = (
        f'Serialize variables with FP16: {self.fp16} and safetensors: {self.use_safetensors}.'
        )
    operations = []
    for path in modified_paths:
        operations.append(CommitOperationAdd(path_in_repo='/'.join(path.
            split('/')[4:]), path_or_fileobj=path))
    commit_description = (
        "Variables converted by the [`diffusers`' `fp16_safetensors` CLI](https://github.com/huggingface/diffusers/blob/main/src/diffusers/commands/fp16_safetensors.py)."
        )
    hub_pr_url = create_commit(repo_id=self.ckpt_id, operations=operations,
        commit_message=commit_message, commit_description=
        commit_description, repo_type='model', create_pr=True).pr_url
    self.logger.info(f'PR created here: {hub_pr_url}.')
