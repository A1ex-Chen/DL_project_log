import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from jax import pmap

# Let's cache the model compilation, so that it doesn't take as long the next time around.
from jax.experimental.compilation_cache import compilation_cache as cc

from diffusers import FlaxStableDiffusionXLPipeline


cc.initialize_cache("/tmp/sdxl_cache")


NUM_DEVICES = jax.device_count()

# 1. Let's start by downloading the model and loading it into our pipeline class
# Adhering to JAX's functional approach, the model's parameters are returned seperatetely and
# will have to be passed to the pipeline during inference
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", revision="refs/pr/95", split_head_dim=True
)

# 2. We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

# 3. Next, we define the different inputs to the pipeline
default_prompt = "a colorful photo of a castle in the middle of a forest with trees and bushes, by Ismail Inceoglu, shadows, high contrast, dynamic shading, hdr, detailed vegetation, digital painting, digital drawing, detailed painting, a detailed digital painting, gothic art, featured on deviantart"
default_neg_prompt = "fog, grainy, purple"
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
width = 1024
height = 1024


# 4. In order to be able to compile the pipeline
# all inputs have to be tensors or strings
# Let's tokenize the prompt and negative prompt


# 5. To make full use of JAX's parallelization capabilities
# the parameters and input tensors are duplicated across devices
# To make sure every device generates a different image, we create
# different seeds for each image. The model parameters won't change
# during inference so we do not wrap them into a function
p_params = replicate(params)




# 6. To compile the pipeline._generate function, we must pass all parameters
# to the function and tell JAX which are static arguments, that is, arguments that
# are known at compile time and won't change. In our case, it is num_inference_steps,
# height, width and return_latents.
# Once the function is compiled, these parameters are ommited from future calls and
# cannot be changed without modifying the code and recompiling.


start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")


# 7. Let's now put it all together in a generate function.


# 8. The first forward pass after AOT compilation still takes a while longer than
# subsequent passes, this is because on the first pass, JAX uses Python dispatch, which
# Fills the C++ dispatch cache.
# When using jit, this extra step is done automatically, but when using AOT compilation,
# it doesn't happen until the function call is made.
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"First inference in {time.time() - start}")

# 9. From this point forward, any calls to generate should result in a faster inference
# time and it won't change.
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")

for i, image in enumerate(images):
    image.save(f"castle_{i}.png")