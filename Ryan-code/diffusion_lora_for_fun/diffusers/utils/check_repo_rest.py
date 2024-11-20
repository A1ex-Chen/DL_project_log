# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

import importlib
import inspect
import os
import re
import warnings
from collections import OrderedDict
from difflib import get_close_matches
from pathlib import Path

from diffusers.models.auto import get_values
from diffusers.utils import ENV_VARS_TRUE_VALUES, is_flax_available, is_torch_available


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_repo.py
PATH_TO_DIFFUSERS = "src/diffusers"
PATH_TO_TESTS = "tests"
PATH_TO_DOC = "docs/source/en"

# Update this list with models that are supposed to be private.
PRIVATE_MODELS = [
    "DPRSpanPredictor",
    "RealmBertModel",
    "T5Stack",
    "TFDPRSpanPredictor",
]

# Update this list for models that are not tested with a comment explaining the reason it should not be.
# Being in this list is an exception and should **not** be the rule.
IGNORE_NON_TESTED = PRIVATE_MODELS.copy() + [
    # models to ignore for not tested
    "OPTDecoder",  # Building part of bigger (tested) model.
    "DecisionTransformerGPT2Model",  # Building part of bigger (tested) model.
    "SegformerDecodeHead",  # Building part of bigger (tested) model.
    "PLBartEncoder",  # Building part of bigger (tested) model.
    "PLBartDecoder",  # Building part of bigger (tested) model.
    "PLBartDecoderWrapper",  # Building part of bigger (tested) model.
    "BigBirdPegasusEncoder",  # Building part of bigger (tested) model.
    "BigBirdPegasusDecoder",  # Building part of bigger (tested) model.
    "BigBirdPegasusDecoderWrapper",  # Building part of bigger (tested) model.
    "DetrEncoder",  # Building part of bigger (tested) model.
    "DetrDecoder",  # Building part of bigger (tested) model.
    "DetrDecoderWrapper",  # Building part of bigger (tested) model.
    "M2M100Encoder",  # Building part of bigger (tested) model.
    "M2M100Decoder",  # Building part of bigger (tested) model.
    "Speech2TextEncoder",  # Building part of bigger (tested) model.
    "Speech2TextDecoder",  # Building part of bigger (tested) model.
    "LEDEncoder",  # Building part of bigger (tested) model.
    "LEDDecoder",  # Building part of bigger (tested) model.
    "BartDecoderWrapper",  # Building part of bigger (tested) model.
    "BartEncoder",  # Building part of bigger (tested) model.
    "BertLMHeadModel",  # Needs to be setup as decoder.
    "BlenderbotSmallEncoder",  # Building part of bigger (tested) model.
    "BlenderbotSmallDecoderWrapper",  # Building part of bigger (tested) model.
    "BlenderbotEncoder",  # Building part of bigger (tested) model.
    "BlenderbotDecoderWrapper",  # Building part of bigger (tested) model.
    "MBartEncoder",  # Building part of bigger (tested) model.
    "MBartDecoderWrapper",  # Building part of bigger (tested) model.
    "MegatronBertLMHeadModel",  # Building part of bigger (tested) model.
    "MegatronBertEncoder",  # Building part of bigger (tested) model.
    "MegatronBertDecoder",  # Building part of bigger (tested) model.
    "MegatronBertDecoderWrapper",  # Building part of bigger (tested) model.
    "PegasusEncoder",  # Building part of bigger (tested) model.
    "PegasusDecoderWrapper",  # Building part of bigger (tested) model.
    "DPREncoder",  # Building part of bigger (tested) model.
    "ProphetNetDecoderWrapper",  # Building part of bigger (tested) model.
    "RealmBertModel",  # Building part of bigger (tested) model.
    "RealmReader",  # Not regular model.
    "RealmScorer",  # Not regular model.
    "RealmForOpenQA",  # Not regular model.
    "ReformerForMaskedLM",  # Needs to be setup as decoder.
    "Speech2Text2DecoderWrapper",  # Building part of bigger (tested) model.
    "TFDPREncoder",  # Building part of bigger (tested) model.
    "TFElectraMainLayer",  # Building part of bigger (tested) model (should it be a TFModelMixin ?)
    "TFRobertaForMultipleChoice",  # TODO: fix
    "TrOCRDecoderWrapper",  # Building part of bigger (tested) model.
    "SeparableConv1D",  # Building part of bigger (tested) model.
    "FlaxBartForCausalLM",  # Building part of bigger (tested) model.
    "FlaxBertForCausalLM",  # Building part of bigger (tested) model. Tested implicitly through FlaxRobertaForCausalLM.
    "OPTDecoderWrapper",
]

# Update this list with test files that don't have a tester with a `all_model_classes` variable and which don't
# trigger the common tests.
TEST_FILES_WITH_NO_COMMON_TESTS = [
    "models/decision_transformer/test_modeling_decision_transformer.py",
    "models/camembert/test_modeling_camembert.py",
    "models/mt5/test_modeling_flax_mt5.py",
    "models/mbart/test_modeling_mbart.py",
    "models/mt5/test_modeling_mt5.py",
    "models/pegasus/test_modeling_pegasus.py",
    "models/camembert/test_modeling_tf_camembert.py",
    "models/mt5/test_modeling_tf_mt5.py",
    "models/xlm_roberta/test_modeling_tf_xlm_roberta.py",
    "models/xlm_roberta/test_modeling_flax_xlm_roberta.py",
    "models/xlm_prophetnet/test_modeling_xlm_prophetnet.py",
    "models/xlm_roberta/test_modeling_xlm_roberta.py",
    "models/vision_text_dual_encoder/test_modeling_vision_text_dual_encoder.py",
    "models/vision_text_dual_encoder/test_modeling_flax_vision_text_dual_encoder.py",
    "models/decision_transformer/test_modeling_decision_transformer.py",
]

# Update this list for models that are not in any of the auto MODEL_XXX_MAPPING. Being in this list is an exception and
# should **not** be the rule.
IGNORE_NON_AUTO_CONFIGURED = PRIVATE_MODELS.copy() + [
    # models to ignore for model xxx mapping
    "DPTForDepthEstimation",
    "DecisionTransformerGPT2Model",
    "GLPNForDepthEstimation",
    "ViltForQuestionAnswering",
    "ViltForImagesAndTextClassification",
    "ViltForImageAndTextRetrieval",
    "ViltForMaskedLM",
    "XGLMEncoder",
    "XGLMDecoder",
    "XGLMDecoderWrapper",
    "PerceiverForMultimodalAutoencoding",
    "PerceiverForOpticalFlow",
    "SegformerDecodeHead",
    "FlaxBeitForMaskedImageModeling",
    "PLBartEncoder",
    "PLBartDecoder",
    "PLBartDecoderWrapper",
    "BeitForMaskedImageModeling",
    "CLIPTextModel",
    "CLIPVisionModel",
    "TFCLIPTextModel",
    "TFCLIPVisionModel",
    "FlaxCLIPTextModel",
    "FlaxCLIPVisionModel",
    "FlaxWav2Vec2ForCTC",
    "DetrForSegmentation",
    "DPRReader",
    "FlaubertForQuestionAnswering",
    "FlavaImageCodebook",
    "FlavaTextModel",
    "FlavaImageModel",
    "FlavaMultimodalModel",
    "GPT2DoubleHeadsModel",
    "LukeForMaskedLM",
    "LukeForEntityClassification",
    "LukeForEntityPairClassification",
    "LukeForEntitySpanClassification",
    "OpenAIGPTDoubleHeadsModel",
    "RagModel",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
    "RealmEmbedder",
    "RealmForOpenQA",
    "RealmScorer",
    "RealmReader",
    "TFDPRReader",
    "TFGPT2DoubleHeadsModel",
    "TFOpenAIGPTDoubleHeadsModel",
    "TFRagModel",
    "TFRagSequenceForGeneration",
    "TFRagTokenForGeneration",
    "Wav2Vec2ForCTC",
    "HubertForCTC",
    "SEWForCTC",
    "SEWDForCTC",
    "XLMForQuestionAnswering",
    "XLNetForQuestionAnswering",
    "SeparableConv1D",
    "VisualBertForRegionToPhraseAlignment",
    "VisualBertForVisualReasoning",
    "VisualBertForQuestionAnswering",
    "VisualBertForMultipleChoice",
    "TFWav2Vec2ForCTC",
    "TFHubertForCTC",
    "MaskFormerForInstanceSegmentation",
]

# Update this list for models that have multiple model types for the same
# model doc
MODEL_TYPE_TO_DOC_MAPPING = OrderedDict(
    [
        ("data2vec-text", "data2vec"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-vision", "data2vec"),
    ]
)


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "diffusers",
    os.path.join(PATH_TO_DIFFUSERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_DIFFUSERS],
)
diffusers = spec.loader.load_module()




# If some modeling modules should be ignored for all checks, they should be added in the nested list
# _ignore_modules of this function.








# If some test_modeling files should be ignored when checking models are all tested, they should be added in the
# nested list _ignore_files of this function.


# This is a bit hacky but I didn't find a way to import the test_file as a module and read inside the tester class
# for the all_model_classes variable.














_re_decorator = re.compile(r"^\s*@(\S+)\s+$")








# One good reason for not being documented is to be deprecated. Put in this list deprecated objects.
DEPRECATED_OBJECTS = [
    "AutoModelWithLMHead",
    "BartPretrainedModel",
    "DataCollator",
    "DataCollatorForSOP",
    "GlueDataset",
    "GlueDataTrainingArguments",
    "LineByLineTextDataset",
    "LineByLineWithRefDataset",
    "LineByLineWithSOPTextDataset",
    "PretrainedBartModel",
    "PretrainedFSMTModel",
    "SingleSentenceClassificationProcessor",
    "SquadDataTrainingArguments",
    "SquadDataset",
    "SquadExample",
    "SquadFeatures",
    "SquadV1Processor",
    "SquadV2Processor",
    "TFAutoModelWithLMHead",
    "TFBartPretrainedModel",
    "TextDataset",
    "TextDatasetForNextSentencePrediction",
    "Wav2Vec2ForMaskedLM",
    "Wav2Vec2Tokenizer",
    "glue_compute_metrics",
    "glue_convert_examples_to_features",
    "glue_output_modes",
    "glue_processors",
    "glue_tasks_num_labels",
    "squad_convert_examples_to_features",
    "xnli_compute_metrics",
    "xnli_output_modes",
    "xnli_processors",
    "xnli_tasks_num_labels",
    "TFTrainer",
    "TFTrainingArguments",
]

# Exceptionally, some objects should not be documented after all rules passed.
# ONLY PUT SOMETHING IN THIS LIST AS A LAST RESORT!
UNDOCUMENTED_OBJECTS = [
    "AddedToken",  # This is a tokenizers class.
    "BasicTokenizer",  # Internal, should never have been in the main init.
    "CharacterTokenizer",  # Internal, should never have been in the main init.
    "DPRPretrainedReader",  # Like an Encoder.
    "DummyObject",  # Just picked by mistake sometimes.
    "MecabTokenizer",  # Internal, should never have been in the main init.
    "ModelCard",  # Internal type.
    "SqueezeBertModule",  # Internal building block (should have been called SqueezeBertLayer)
    "TFDPRPretrainedReader",  # Like an Encoder.
    "TransfoXLCorpus",  # Internal type.
    "WordpieceTokenizer",  # Internal, should never have been in the main init.
    "absl",  # External module
    "add_end_docstrings",  # Internal, should never have been in the main init.
    "add_start_docstrings",  # Internal, should never have been in the main init.
    "cached_path",  # Internal used for downloading models.
    "convert_tf_weight_name_to_pt_weight_name",  # Internal used to convert model weights
    "logger",  # Internal logger
    "logging",  # External module
    "requires_backends",  # Internal function
]

# This list should be empty. Objects in it should get their own doc page.
SHOULD_HAVE_THEIR_OWN_PAGE = [
    # Benchmarks
    "PyTorchBenchmark",
    "PyTorchBenchmarkArguments",
    "TensorFlowBenchmark",
    "TensorFlowBenchmarkArguments",
]








# Re pattern to catch :obj:`xx`, :class:`xx`, :func:`xx` or :meth:`xx`.
_re_rst_special_words = re.compile(r":(?:obj|func|class|meth):`([^`]+)`")
# Re pattern to catch things between double backquotes.
_re_double_backquotes = re.compile(r"(^|[^`])``([^`]+)``([^`]|$)")
# Re pattern to catch example introduction.
_re_rst_example = re.compile(r"^\s*Example.*::\s*$", flags=re.MULTILINE)








if __name__ == "__main__":
    check_repo_quality()