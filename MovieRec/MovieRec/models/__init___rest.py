from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel
}

