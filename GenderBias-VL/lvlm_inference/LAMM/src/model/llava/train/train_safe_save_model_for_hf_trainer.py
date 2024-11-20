def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
    output_dir: str):
    """Collects the state dict and dump to disk."""
    if getattr(trainer.args, 'tune_mm_mlp_adapter', False):
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, 'use_im_start_end', False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.
            named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder,
                    'mm_projector')
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder,
                    f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir,
                    f'mm_projector.bin'))
        return
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()
            }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
