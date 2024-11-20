# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RAG Retriever model implementation."""

import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ...file_utils import (
    cached_path,
    is_datasets_available,
    is_faiss_available,
    is_remote_url,
    requires_datasets,
    requires_faiss,
)
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer


if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss


logger = logging.get_logger(__name__)


LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"


class Index:
    """
    A base class for the Indices encapsulated by the :class:`~transformers.RagRetriever`.
    """






class LegacyIndex(Index):
    """
    An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR. We use
    default faiss index parameters as specified in that repository.

    Args:
        vector_size (:obj:`int`):
            The dimension of indexed vectors.
        index_path (:obj:`str`):
            A path to a `directory` containing index files compatible with
            :class:`~transformers.models.rag.retrieval_rag.LegacyIndex`
    """

    INDEX_FILENAME = "hf_bert_base.hnswSQ8_correct_phi_128.c_index"
    PASSAGE_FILENAME = "psgs_w100.tsv.pkl"










class HFIndexBase(Index):







class CanonicalHFIndex(HFIndexBase):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. If ``index_path`` is set to ``None``, we load the
    pre-computed index available with the :class:`~datasets.arrow_dataset.Dataset`, otherwise, we load the index from
    the indicated path on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_name (:obj:`str`, optional, defaults to ``wiki_dpr``):
            A datatset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids
            with ``datasets.list_datasets()``).
        dataset_split (:obj:`str`, optional, defaults to ``train``)
            Which split of the ``dataset`` to load.
        index_name (:obj:`str`, optional, defaults to ``train``)
            The index_name of the index associated with the ``dataset``. The index loaded from ``index_path`` will be
            saved under this name.
        index_path (:obj:`str`, optional, defaults to ``None``)
            The path to the serialized faiss index on disk.
        use_dummy_dataset (:obj:`bool`, optional, defaults to ``False``): If True, use the dummy configuration of the dataset for tests.
    """




class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. The dataset and the index are both loaded from the
    indicated paths on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_path (:obj:`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (:obj:`str`)
            The path to the serialized faiss index on disk.
    """


    @classmethod



class RagRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            ``Index`` to build. You can load your own custom dataset with ``config.index_name="custom"`` or use a
            canonical one (default) from the datasets library with ``config.index_name="wiki_dpr"`` for example.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        index (:class:`~transformers.models.rag.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples::

        >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
        >>> from transformers import RagRetriever
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', dataset="wiki_dpr", index_name='compressed')

        >>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
        >>> from transformers import RagRetriever
        >>> dataset = ...  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', indexed_dataset=dataset)

        >>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
        >>> from transformers import RagRetriever
        >>> dataset_path = "path/to/my/dataset"  # dataset saved via `dataset.save_to_disk(...)`
        >>> index_path = "path/to/my/index.faiss"  # faiss index saved via `dataset.get_index("embeddings").save(...)`
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', index_name='custom', passages_path=dataset_path, index_path=index_path)

        >>> # To load the legacy index built originally for Rag's paper
        >>> from transformers import RagRetriever
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', index_name='legacy')

    """

    _init_retrieval = True


    @staticmethod

    @classmethod








        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            logger.debug(
                "index search time: {} sec, batch size {}".format(
                    time.time() - start_time, question_hidden_states.shape
                )
            )
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The retrieval
              embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Returns: :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following
        fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": retrieved_doc_embeds,
                "doc_ids": doc_ids,
            },
            tensor_type=return_tensors,
        )