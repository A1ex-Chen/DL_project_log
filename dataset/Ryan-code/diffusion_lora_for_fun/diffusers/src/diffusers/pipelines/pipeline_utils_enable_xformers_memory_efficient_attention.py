def enable_xformers_memory_efficient_attention(self, attention_op: Optional
    [Callable]=None):
    """
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
        option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
        up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        >>> # Workaround for not accepting attention shape using VAE for Flash Attention
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
    self.set_use_memory_efficient_attention_xformers(True, attention_op)
