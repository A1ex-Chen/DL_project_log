import torch
import soundfile as sf
import numpy as np
import random, os

from consistencytta import ConsistencyTTA






device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)
sr = 16000

# Build ConsistencyTTA model
consistencytta = ConsistencyTTA().to(device)
consistencytta.eval()
consistencytta.requires_grad_(False)

# Generate audio (feel free to change the seed and prompt)
print("Generating test audio for prompt 'A dog barks as a train passes by.'...")
generate("A dog barks as a train passes by.", seed=1)

while True:
    prompt = input("Enter a prompt: ")
    generate(prompt)
    print(f"Audio generated successfully for prompt {prompt}!")