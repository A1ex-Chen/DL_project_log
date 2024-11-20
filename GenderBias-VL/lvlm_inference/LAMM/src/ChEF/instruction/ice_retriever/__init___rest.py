from .random_retriever import RandomRetriever
from .topk_retriever import TopkRetriever
from .fixed_retriever import FixedRetriever
from .topk_retriever_img import ImageTopkRetriever


retriever_dict = {
    'random': RandomRetriever,
    'topk_text': TopkRetriever,
    'fixed': FixedRetriever,
    'topk_img': ImageTopkRetriever
}
