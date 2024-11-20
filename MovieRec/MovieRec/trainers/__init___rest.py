from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer
}

