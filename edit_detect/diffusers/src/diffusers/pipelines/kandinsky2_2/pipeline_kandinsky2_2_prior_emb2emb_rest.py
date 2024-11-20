from typing import List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import PriorTransformer
from ...schedulers import UnCLIPScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..kandinsky import KandinskyPriorPipelineOutput
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorEmb2EmbPipeline
        >>> import torch

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> prompt = "red cat, 4k photo"
        >>> img = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )
        >>> image_emb, nagative_image_emb = pipe_prior(prompt, image=img, strength=0.2).to_tuple()

        >>> pipe = KandinskyPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder, torch_dtype=torch.float16"
        ... )
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ... ).images

        >>> image[0].save("cat.png")
        ```
"""

EXAMPLE_INTERPOLATE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22Pipeline
        >>> from diffusers.utils import load_image
        >>> import PIL

        >>> import torch
        >>> from torchvision import transforms

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> img1 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        >>> img2 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/starry_night.jpeg"
        ... )

        >>> images_texts = ["a cat", img1, img2]
        >>> weights = [0.3, 0.3, 0.4]
        >>> image_emb, zero_image_emb = pipe_prior.interpolate(images_texts, weights)

        >>> pipe = KandinskyV22Pipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=150,
        ... ).images[0]

        >>> image.save("starry_cat.png")
        ```
"""


class KandinskyV22PriorEmb2EmbPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->prior"
    _exclude_from_cpu_offload = ["prior"]



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)



    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline.get_zero_embed

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline._encode_prompt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)