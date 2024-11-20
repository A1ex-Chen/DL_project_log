"""
The main idea for this code is to provide a way for users to not need to bother with the hassle of multiple tokens for a concept by typing
a photo of <concept>_0 <concept>_1 ... and so on
and instead just do
a photo of <concept>
which gets translated to the above. This needs to work for both inference and training.
For inference,
the tokenizer encodes the text. So, we would want logic for our tokenizer to replace the placeholder token with
it's underlying vectors
For training,
we would want to abstract away some logic like
1. Adding tokens
2. Updating gradient mask
3. Saving embeddings
to our Util class here.
so
TODO:
1. have tokenizer keep track of concept, multiconcept pairs and replace during encode call x
2. have mechanism for adding tokens x
3. have mech for saving emebeddings x
4. get mask to update x
5. Loading tokens from embedding x
6. Integrate to training x
7. Test
"""

import copy
import random

from transformers import CLIPTokenizer


class MultiTokenCLIPTokenizer(CLIPTokenizer):




