# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Convert pytorch checkpoints to TensorFlow """


import argparse
import os

from transformers import (
    ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
    DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
    DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    WEIGHTS_NAME,
    XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AlbertConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    FlaubertConfig,
    GPT2Config,
    LxmertConfig,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TFAlbertForPreTraining,
    TFBartForConditionalGeneration,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFCamembertForMaskedLM,
    TFCTRLLMHeadModel,
    TFDistilBertForMaskedLM,
    TFDistilBertForQuestionAnswering,
    TFDPRContextEncoder,
    TFDPRQuestionEncoder,
    TFDPRReader,
    TFElectraForPreTraining,
    TFFlaubertWithLMHeadModel,
    TFGPT2LMHeadModel,
    TFLxmertForPreTraining,
    TFLxmertVisualFeatureEncoder,
    TFOpenAIGPTLMHeadModel,
    TFRobertaForMaskedLM,
    TFRobertaForSequenceClassification,
    TFT5ForConditionalGeneration,
    TFTransfoXLLMHeadModel,
    TFXLMRobertaForMaskedLM,
    TFXLMWithLMHeadModel,
    TFXLNetLMHeadModel,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    cached_path,
    is_torch_available,
    load_pytorch_checkpoint_in_tf2_model,
)
from transformers.file_utils import hf_bucket_url
from transformers.utils import logging


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        AlbertForPreTraining,
        BartForConditionalGeneration,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        CamembertForMaskedLM,
        CTRLLMHeadModel,
        DistilBertForMaskedLM,
        DistilBertForQuestionAnswering,
        DPRContextEncoder,
        DPRQuestionEncoder,
        DPRReader,
        ElectraForPreTraining,
        FlaubertWithLMHeadModel,
        GPT2LMHeadModel,
        LxmertForPreTraining,
        LxmertVisualFeatureEncoder,
        OpenAIGPTLMHeadModel,
        RobertaForMaskedLM,
        RobertaForSequenceClassification,
        T5ForConditionalGeneration,
        TransfoXLLMHeadModel,
        XLMRobertaForMaskedLM,
        XLMWithLMHeadModel,
        XLNetLMHeadModel,
    )


logging.set_verbosity_info()

MODEL_CLASSES = {
    "bart": (
        BartConfig,
        TFBartForConditionalGeneration,
        BartForConditionalGeneration,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "bert": (
        BertConfig,
        TFBertForPreTraining,
        BertForPreTraining,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-large-cased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-base-cased-finetuned-mrpc": (
        BertConfig,
        TFBertForSequenceClassification,
        BertForSequenceClassification,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "dpr": (
        DPRConfig,
        TFDPRQuestionEncoder,
        TFDPRContextEncoder,
        TFDPRReader,
        DPRQuestionEncoder,
        DPRContextEncoder,
        DPRReader,
        DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "gpt2": (
        GPT2Config,
        TFGPT2LMHeadModel,
        GPT2LMHeadModel,
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlnet": (
        XLNetConfig,
        TFXLNetLMHeadModel,
        XLNetLMHeadModel,
        XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlm": (
        XLMConfig,
        TFXLMWithLMHeadModel,
        XLMWithLMHeadModel,
        XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlm-roberta": (
        XLMRobertaConfig,
        TFXLMRobertaForMaskedLM,
        XLMRobertaForMaskedLM,
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "transfo-xl": (
        TransfoXLConfig,
        TFTransfoXLLMHeadModel,
        TransfoXLLMHeadModel,
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "openai-gpt": (
        OpenAIGPTConfig,
        TFOpenAIGPTLMHeadModel,
        OpenAIGPTLMHeadModel,
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "roberta": (
        RobertaConfig,
        TFRobertaForMaskedLM,
        RobertaForMaskedLM,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "roberta-large-mnli": (
        RobertaConfig,
        TFRobertaForSequenceClassification,
        RobertaForSequenceClassification,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "camembert": (
        CamembertConfig,
        TFCamembertForMaskedLM,
        CamembertForMaskedLM,
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "flaubert": (
        FlaubertConfig,
        TFFlaubertWithLMHeadModel,
        FlaubertWithLMHeadModel,
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "distilbert": (
        DistilBertConfig,
        TFDistilBertForMaskedLM,
        DistilBertForMaskedLM,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "distilbert-base-distilled-squad": (
        DistilBertConfig,
        TFDistilBertForQuestionAnswering,
        DistilBertForQuestionAnswering,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "lxmert": (
        LxmertConfig,
        TFLxmertForPreTraining,
        LxmertForPreTraining,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "lxmert-visual-feature-encoder": (
        LxmertConfig,
        TFLxmertVisualFeatureEncoder,
        LxmertVisualFeatureEncoder,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "ctrl": (
        CTRLConfig,
        TFCTRLLMHeadModel,
        CTRLLMHeadModel,
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "albert": (
        AlbertConfig,
        TFAlbertForPreTraining,
        AlbertForPreTraining,
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "t5": (
        T5Config,
        TFT5ForConditionalGeneration,
        T5ForConditionalGeneration,
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "electra": (
        ElectraConfig,
        TFElectraForPreTraining,
        ElectraForPreTraining,
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
}






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_dump_path", default=None, type=str, required=True, help="Path to the output Tensorflow dump file."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help="Model type selected in the list of {}. If not given, will download and convert all the models from AWS.".format(
            list(MODEL_CLASSES.keys())
        ),
    )
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default=None,
        type=str,
        help="Path to the PyTorch checkpoint path or shortcut name to download from AWS. "
        "If not given, will download and convert all the checkpoints from AWS.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="The config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture. If not given and "
        "--pytorch_checkpoint_path is not given or is a shortcut name"
        "use the configuration associated to the shortcut name on the AWS",
    )
    parser.add_argument(
        "--compare_with_pt_model", action="store_true", help="Compare Tensorflow and PyTorch model predictions."
    )
    parser.add_argument(
        "--use_cached_models",
        action="store_true",
        help="Use cached models if possible instead of updating to latest checkpoint versions.",
    )
    parser.add_argument(
        "--remove_cached_files",
        action="store_true",
        help="Remove pytorch models after conversion (save memory when converting in batches).",
    )
    parser.add_argument("--only_convert_finetuned_models", action="store_true", help="Only convert finetuned models.")
    args = parser.parse_args()

    # if args.pytorch_checkpoint_path is not None:
    #     convert_pt_checkpoint_to_tf(args.model_type.lower(),
    #                                 args.pytorch_checkpoint_path,
    #                                 args.config_file if args.config_file is not None else args.pytorch_checkpoint_path,
    #                                 args.tf_dump_path,
    #                                 compare_with_pt_model=args.compare_with_pt_model,
    #                                 use_cached_models=args.use_cached_models)
    # else:
    convert_all_pt_checkpoints_to_tf(
        args.model_type.lower() if args.model_type is not None else None,
        args.tf_dump_path,
        model_shortcut_names_or_path=[args.pytorch_checkpoint_path]
        if args.pytorch_checkpoint_path is not None
        else None,
        config_shortcut_names_or_path=[args.config_file] if args.config_file is not None else None,
        compare_with_pt_model=args.compare_with_pt_model,
        use_cached_models=args.use_cached_models,
        remove_cached_files=args.remove_cached_files,
        only_convert_finetuned_models=args.only_convert_finetuned_models,
    )