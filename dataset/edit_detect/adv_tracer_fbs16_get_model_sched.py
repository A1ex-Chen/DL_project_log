@staticmethod
def get_model_sched(model_type: str, model_id: str):
    vae = None
    pipe: DiffusionPipeline = ConsistencyModelPipeline.from_pretrained(
        ModelSched.get_md_id(model_id), torch_dtype=torch.float16)
    return pipe, pipe.unet.eval(), vae, pipe.scheduler
