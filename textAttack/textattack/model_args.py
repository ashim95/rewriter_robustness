"""
ModelArgs Class
===============
"""


from ast import Name
from dataclasses import dataclass
import json
import os
from random import sample

import transformers

import textattack
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file
from textattack.robustLLM import get_default_cls_model_args, get_default_interceptor_args
from textattack.robustLLM import FullModel, FullRandomModel
# from textattack.robustLLM import AdfarModel
from textattack.robustLLM import ShieldModel
from textattack.robustLLM import SaferModel
from textattack.robustLLM import SampleShieldModelWrapper
from argparse import Namespace
import torch

HUGGINGFACE_MODELS = {
    #
    # bert-base-uncased
    #
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-uncased-ag-news": "textattack/bert-base-uncased-ag-news",
    "bert-base-uncased-cola": "textattack/bert-base-uncased-CoLA",
    "bert-base-uncased-imdb": "textattack/bert-base-uncased-imdb",
    "bert-base-uncased-mnli": "textattack/bert-base-uncased-MNLI",
    "bert-base-uncased-mrpc": "textattack/bert-base-uncased-MRPC",
    "bert-base-uncased-qnli": "textattack/bert-base-uncased-QNLI",
    "bert-base-uncased-qqp": "textattack/bert-base-uncased-QQP",
    "bert-base-uncased-rte": "textattack/bert-base-uncased-RTE",
    "bert-base-uncased-sst2": "textattack/bert-base-uncased-SST-2",
    "bert-base-uncased-stsb": "textattack/bert-base-uncased-STS-B",
    "bert-base-uncased-wnli": "textattack/bert-base-uncased-WNLI",
    "bert-base-uncased-mr": "textattack/bert-base-uncased-rotten-tomatoes",
    "bert-base-uncased-snli": "textattack/bert-base-uncased-snli",
    "bert-base-uncased-yelp": "textattack/bert-base-uncased-yelp-polarity",
    #
    # distilbert-base-cased
    #
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilbert-base-cased-cola": "textattack/distilbert-base-cased-CoLA",
    "distilbert-base-cased-mrpc": "textattack/distilbert-base-cased-MRPC",
    "distilbert-base-cased-qqp": "textattack/distilbert-base-cased-QQP",
    "distilbert-base-cased-snli": "textattack/distilbert-base-cased-snli",
    "distilbert-base-cased-sst2": "textattack/distilbert-base-cased-SST-2",
    "distilbert-base-cased-stsb": "textattack/distilbert-base-cased-STS-B",
    "distilbert-base-uncased-ag-news": "textattack/distilbert-base-uncased-ag-news",
    "distilbert-base-uncased-cola": "textattack/distilbert-base-cased-CoLA",
    "distilbert-base-uncased-imdb": "textattack/distilbert-base-uncased-imdb",
    "distilbert-base-uncased-mnli": "textattack/distilbert-base-uncased-MNLI",
    "distilbert-base-uncased-mr": "textattack/distilbert-base-uncased-rotten-tomatoes",
    "distilbert-base-uncased-mrpc": "textattack/distilbert-base-uncased-MRPC",
    "distilbert-base-uncased-qnli": "textattack/distilbert-base-uncased-QNLI",
    "distilbert-base-uncased-rte": "textattack/distilbert-base-uncased-RTE",
    "distilbert-base-uncased-wnli": "textattack/distilbert-base-uncased-WNLI",
    #
    # roberta-base (RoBERTa is cased by default)
    #
    "roberta-base": "roberta-base",
    "roberta-base-ag-news": "textattack/roberta-base-ag-news",
    "roberta-base-cola": "textattack/roberta-base-CoLA",
    "roberta-base-imdb": "textattack/roberta-base-imdb",
    "roberta-base-mr": "textattack/roberta-base-rotten-tomatoes",
    "roberta-base-mrpc": "textattack/roberta-base-MRPC",
    "roberta-base-qnli": "textattack/roberta-base-QNLI",
    "roberta-base-rte": "textattack/roberta-base-RTE",
    "roberta-base-sst2": "textattack/roberta-base-SST-2",
    "roberta-base-stsb": "textattack/roberta-base-STS-B",
    "roberta-base-wnli": "textattack/roberta-base-WNLI",
    #
    # albert-base-v2 (ALBERT is cased by default)
    #
    "albert-base-v2": "albert-base-v2",
    "albert-base-v2-ag-news": "textattack/albert-base-v2-ag-news",
    "albert-base-v2-cola": "textattack/albert-base-v2-CoLA",
    "albert-base-v2-imdb": "textattack/albert-base-v2-imdb",
    "albert-base-v2-mr": "textattack/albert-base-v2-rotten-tomatoes",
    "albert-base-v2-rte": "textattack/albert-base-v2-RTE",
    "albert-base-v2-qqp": "textattack/albert-base-v2-QQP",
    "albert-base-v2-snli": "textattack/albert-base-v2-snli",
    "albert-base-v2-sst2": "textattack/albert-base-v2-SST-2",
    "albert-base-v2-stsb": "textattack/albert-base-v2-STS-B",
    "albert-base-v2-wnli": "textattack/albert-base-v2-WNLI",
    "albert-base-v2-yelp": "textattack/albert-base-v2-yelp-polarity",
    #
    # xlnet-base-cased
    #
    "xlnet-base-cased": "xlnet-base-cased",
    "xlnet-base-cased-cola": "textattack/xlnet-base-cased-CoLA",
    "xlnet-base-cased-imdb": "textattack/xlnet-base-cased-imdb",
    "xlnet-base-cased-mr": "textattack/xlnet-base-cased-rotten-tomatoes",
    "xlnet-base-cased-mrpc": "textattack/xlnet-base-cased-MRPC",
    "xlnet-base-cased-rte": "textattack/xlnet-base-cased-RTE",
    "xlnet-base-cased-stsb": "textattack/xlnet-base-cased-STS-B",
    "xlnet-base-cased-wnli": "textattack/xlnet-base-cased-WNLI",
}


#
# Models hosted by textattack.
# `models` vs `models_v2`: `models_v2` is simply a new dir in S3 that contains models' `config.json`.
# Fixes issue https://github.com/QData/TextAttack/issues/485
# Model parameters has not changed.
#
TEXTATTACK_MODELS = {
    #
    # LSTMs
    #
    "lstm-ag-news": "models_v2/classification/lstm/ag-news",
    "lstm-imdb": "models_v2/classification/lstm/imdb",
    "lstm-mr": "models_v2/classification/lstm/mr",
    "lstm-sst2": "models_v2/classification/lstm/sst2",
    "lstm-yelp": "models_v2/classification/lstm/yelp",
    #
    # CNNs
    #
    "cnn-ag-news": "models_v2/classification/cnn/ag-news",
    "cnn-imdb": "models_v2/classification/cnn/imdb",
    "cnn-mr": "models_v2/classification/cnn/rotten-tomatoes",
    "cnn-sst2": "models_v2/classification/cnn/sst",
    "cnn-yelp": "models_v2/classification/cnn/yelp",
    #
    # T5 for translation
    #
    "t5-en-de": "english_to_german",
    "t5-en-fr": "english_to_french",
    "t5-en-ro": "english_to_romanian",
    #
    # T5 for summarization
    #
    "t5-summarization": "summarization",
}


@dataclass
class ModelArgs:
    """Arguments for loading base/pretrained or trained models."""

    model: str = None
    model_from_file: str = None
    model_from_huggingface: str = None
    model_type_to_evaluate: str = None
    t5_model_to_use: str = None
    other_args: str = None

    @classmethod
    def _add_parser_args(cls, parser):
        """Adds model-related arguments to an argparser."""
        # model_group = parser.add_mutually_exclusive_group()
        model_group = parser.add_argument_group()

        model_names = list(HUGGINGFACE_MODELS.keys()) + list(TEXTATTACK_MODELS.keys())
        model_group.add_argument(
            "--model",
            type=str,
            required=False,
            default=None,
            help="Name of or path to a pre-trained TextAttack model to load. Choices: "
            + str(model_names),
        )
        model_group.add_argument(
            "--model-from-file",
            type=str,
            required=False,
            help="File of model and tokenizer to import.",
        )
        model_group.add_argument(
            "--model-from-huggingface",
            type=str,
            required=False,
            help="Name of or path of pre-trained HuggingFace model to load.",
        )

        model_group.add_argument(
            "--model-type-to-evaluate",
            type=str,
            default="hf_model",
            required=False,
            help="Which model type to evaluate - A HF model Checkpoint (hf_model), ADFAR model (adfar), Model with T5 as interceptor (t5), shield, safer",
        )

        model_group.add_argument(
            "--t5-model-to-use",
            type=str,
            required=False,
            help="T5 model_path to use when using an interceptor model",
            default='../transformers/examples/pytorch/summarization/t5_base_bert-base-uncased-textfooler_both_adv_orig_multiple'
        )

        model_group.add_argument(
            "--other-args",
            type=str,
            required=False,
            help="Pass other arguments as json parsable string",
            default="{}",
        )


        return parser

    @classmethod
    def _create_model_from_args(cls, args):
        """Given ``ModelArgs``, return specified
        ``textattack.models.wrappers.ModelWrapper`` object."""

        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if args.model_from_file:
            # Support loading the model from a .py file where a model wrapper
            # is instantiated.
            colored_model_name = textattack.shared.utils.color_text(
                args.model_from_file, color="blue", method="ansi"
            )
            textattack.shared.logger.info(
                f"Loading model and tokenizer from file: {colored_model_name}"
            )
            if ARGS_SPLIT_TOKEN in args.model_from_file:
                model_file, model_name = args.model_from_file.split(ARGS_SPLIT_TOKEN)
            else:
                _, model_name = args.model_from_file, "model"
            try:
                model_module = load_module_from_file(args.model_from_file)
            except Exception:
                raise ValueError(f"Failed to import file {args.model_from_file}.")
            try:
                model = getattr(model_module, model_name)
            except AttributeError:
                raise AttributeError(
                    f"Variable `{model_name}` not found in module {args.model_from_file}."
                )

            if not isinstance(model, textattack.models.wrappers.ModelWrapper):
                raise TypeError(
                    f"Variable `{model_name}` must be of type "
                    f"``textattack.models.ModelWrapper``, got type {type(model)}."
                )
        elif (args.model in HUGGINGFACE_MODELS) or args.model_from_huggingface:
            # Support loading models automatically from the HuggingFace model hub.

            if args.other_args is not None:
                args.other_args = json.loads(args.other_args)

            print(args.other_args)
            if args.model_type_to_evaluate.lower() == 'hf_model':

                model_name = (
                    HUGGINGFACE_MODELS[args.model]
                    if (args.model in HUGGINGFACE_MODELS)
                    else args.model_from_huggingface
                )
                colored_model_name = textattack.shared.utils.color_text(
                    model_name, color="blue", method="ansi"
                )
                textattack.shared.logger.info(
                    f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
                )
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name, use_fast=True
                )
                model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

            elif args.model_type_to_evaluate.lower() == 't5':
                # """
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                cls_model_args = get_default_cls_model_args()
                interceptor_args = get_default_interceptor_args()
                # args = Namespace()
                # cls_model_args.hf_model = args.model_from_huggingface
                args.cls_model = "distilbert-base-uncased-finetuned-sst-2-english"
                args.cls_model = "textattack/bert-base-uncased-SST-2"
                # args.cls_model = "textattack/roberta-base-SST-2"
                # args.cls_model = "textattack/distilbert-base-cased-SST-2"
                # args.cls_model = "textattack/albert-base-v2-SST-2"
                # args.cls_model = "textattack/bert-base-uncased-rotten-tomatoes"
                # args.cls_model = "textattack/bert-base-uncased-ag-news"
                # args.cls_model = "distilbert-base-uncased-finetuned-sst-2-english"
                args.cls_model = args.model_from_huggingface
                args.interceptor_model = "random"
                # args.interceptor_model = "../transformers/examples/pytorch/summarization/t5_base_textfooler/"
                args.interceptor_model = "../transformers/examples/pytorch/summarization/t5_base_textfooler_both_adv_orig"
                args.interceptor_model = "../transformers/examples/pytorch/summarization/t5_base_bert-base-uncased-textfooler_both_adv_orig_multiple"
                args.interceptor_model = args.t5_model_to_use
                args.num_labels = 2 # for sst2, mr (rotten tomatoes)
                # args.num_labels = 4 # ag_news
                args.task_name = "sst2"
                # args.task_name = "ag_news"
                args.cls_max_seq_length = 128
                # args.cls_max_seq_length = 512 # ag_news
                args.batch_size = 32
                args.percent_random_defense = 1.0
                args.topk_random_defense = 5
                args.source_prefix = "correct the given sentence: "
                args.max_target_length = 128
                # args.max_target_length = 512 # ag_news
                args.num_beams = 5

                if 'generate_sentence_by_sentence' in args.other_args:
                    if args.other_args['generate_sentence_by_sentence'] == 'true':
                        interceptor_args.generate_sentence_by_sentence = True
                    else:
                        interceptor_args.generate_sentence_by_sentence = False

                if 'generation_method' in args.other_args:
                    interceptor_args.generation_method = args.other_args['generation_method']

                if 'task_name' in args.other_args:
                    args.task_name = args.other_args['task_name']
                    if args.task_name =='ag_news':
                        args.max_target_length = 512
                        args.cls_max_seq_length = 512

                if 'num_labels' in args.other_args:
                    args.num_labels = int(args.other_args['num_labels'])

                if 'source_prefix' in args.other_args:
                    args.source_prefix = args.other_args['source_prefix']

                # print(args)
                # print(interceptor_args)

                if 'ag_news' in args.model_from_huggingface.lower() or 'ag-news' in args.model_from_huggingface.lower():
                    args.num_labels = 4
                    interceptor_args.max_target_length = 512
                    args.cls_max_seq_length = 512
                    args.task_name = "ag_news"
                    interceptor_args.generate_sentence_by_sentence = True

                if not args.interceptor_model.lower() == 'random':
                    interceptor_args.model_name_or_path = args.interceptor_model
                    interceptor_args.source_prefix = args.source_prefix
                    interceptor_args.max_target_length = args.max_target_length
                    #interceptor_args.model_name_or_path = "../transformers/examples/pytorch/summarization/t5_base_textfooler/"

                else:
                    interceptor_args.percent_random_defense = args.percent_random_defense
                    interceptor_args.topk_random_defense = args.topk_random_defense

                cls_model_args.model_name_or_path = args.cls_model
                cls_model_args.num_labels = args.num_labels
                cls_model_args.task_name = args.task_name
                cls_model_args.cls_max_seq_length = args.cls_max_seq_length
                cls_model_args.device= device
                cls_model_args.batch_size = args.batch_size

                interceptor_args.device = device

                # model = FullRandomModel(interceptor_args=interceptor_args, cls_args=cls_model_args)
                model = FullModel(interceptor_args=interceptor_args, cls_args=cls_model_args)
                # """

            elif args.model_type_to_evaluate.lower() == 'shield':

                shield_args = Namespace()
                shield_args.hf_model = args.model_from_huggingface
                shield_args.base_classifier = False
                shield_args.model_type = "bert-base-uncased"
                shield_args.nclasses = 2 # for sst2
                shield_args.temperature = 1.0 # adjust


                if shield_args.base_classifier:
                    shield_args.model_path = "/uufs/chpc.utah.edu/common/home/u1266434/scr/robustness/shield/sst2_base/model.pt"
                    textattack.shared.logger.info(
                    f"Loading pre-trained shield base model from path: {shield_args.model_path}"
                    )
                else:
                    shield_args.model_path = "/uufs/chpc.utah.edu/common/home/u1266434/scr/robustness/shield/sst2_shield/shield.pt"
                    textattack.shared.logger.info(
                    f"Loading pre-trained shield robust model from path: {shield_args.model_path}"
                    )

                shield_args.max_seq_len = 128
                shield_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = ShieldModel(shield_args)

            elif args.model_type_to_evaluate.lower() == 'safer':

                safer_args = Namespace()

                safer_args.model = args.model_from_huggingface
                safer_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                safer_args.num_labels = 2 # imdb, sst2, mr datasets
                safer_args.model_type= "bert"
                safer_args.max_seq_length = 256 # default setting in safer code
                # safer_args.model_name_or_path = "bert-base-uncased" # for loading bert tokenizer
                safer_args.model_name_or_path = args.model_from_huggingface
                safer_args.do_lower_case = True
                safer_args.task_name = "imdb" # or sst2, used for data processor (might not be needed)

                safer_args.similarity_threshold = 0.8
                safer_args.perturbation_constraint = 100
                safer_args.seed = 1 # set random seed for smoothing procedure

                safer_args.word_emb_file = "" # don't meed for evaluation
                safer_args.temp_data_dir = "" # directory where to store intermediate files (not needed)
                safer_args.num_random_samples = 20 # number of perturbations to generate for smoothing
                safer_args.batch_size = 32
                safer_args.perturb_pca_path = "imdb_perturbation_constraint_pca0.8_100.pkl"

                if 'sst2' in args.model_from_huggingface or 'sst-2' in args.model_from_huggingface.lower():
                    safer_args.task_name = "sst2"
                    safer_args.perturb_pca_path = "sst2_perturbation_constraint_pca0.8_100.pkl"
                    safer_args.max_seq_length = 128
                    safer_args.num_labels = 2
                elif 'rotten_tomatoes' in args.model_from_huggingface or 'rotten-tomatoes' in args.model_from_huggingface:
                    safer_args.task_name = "rotten_tomatoes"
                    safer_args.perturb_pca_path = "rotten_tomatoes_perturbation_constraint_pca0.8_100.pkl"
                    safer_args.max_seq_length = 128
                    safer_args.num_labels = 2
                elif 'ag_news' in args.model_from_huggingface or 'ag-news' in args.model_from_huggingface.lower():
                    safer_args.task_name = "ag_news"
                    safer_args.perturb_pca_path = "ag_news_perturbation_constraint_pca0.8_100.pkl"
                    safer_args.max_seq_length = 512
                    safer_args.num_labels = 4

                model = SaferModel(safer_args)
                # print(model.model)

            elif args.model_type_to_evaluate.lower() == "sample":

                sample_args = Namespace()
                sample_args.model = args.model_from_huggingface
                sample_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sample_args.model_type= "bert"
                if 'sst2' in args.model_from_huggingface or 'sst-2' in args.model_from_huggingface.lower():
                    sample_args.task_name = "sst2"
                    sample_args.max_seq_length = 128
                    sample_args.num_labels = 2
                elif 'rotten_tomatoes' in args.model_from_huggingface or 'rotten-tomatoes' in args.model_from_huggingface.lower():
                    sample_args.task_name = "rotten_tomatoes"
                    sample_args.max_seq_length = 128
                    sample_args.num_labels = 2
                elif 'ag_news' in args.model_from_huggingface or 'ag-news' in args.model_from_huggingface.lower():
                    sample_args.task_name = "ag_news"
                    sample_args.max_seq_length = 512
                    sample_args.num_labels = 4
                print('Sample Shielder Arguments are ')
                print(sample_args)
                model = SampleShieldModelWrapper(sample_args)


            elif args.model_type_to_evaluate.lower() == 'disp':

                raise NotImplementedError

            # elif args.model_type_to_evaluate.lower() == 'adfar':
                # # ADFAR Code # requires a different transformers version, so implemented separately
                # """
                # args = Namespace()
                # args.target_model_path = "/uufs/chpc.utah.edu/common/home/u1266434/scr/robustness/adfar/src/experiments/sst2/4times_adv_double_0-7/"
                # args.nclasses = 2
                # args.max_seq_length = 128
                # args.batch_size = 32
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # model = AdfarModel(args)
                # # print(torch.sum(model.model.model.classifier2.weight))
                # # print(torch.sum(model.model.model.classifier1.weight))
                # """

        elif args.model in TEXTATTACK_MODELS:
            # Support loading TextAttack pre-trained models via just a keyword.
            colored_model_name = textattack.shared.utils.color_text(
                args.model, color="blue", method="ansi"
            )
            if args.model.startswith("lstm"):
                textattack.shared.logger.info(
                    f"Loading pre-trained TextAttack LSTM: {colored_model_name}"
                )
                model = textattack.models.helpers.LSTMForClassification.from_pretrained(
                    args.model
                )
            elif args.model.startswith("cnn"):
                textattack.shared.logger.info(
                    f"Loading pre-trained TextAttack CNN: {colored_model_name}"
                )
                model = (
                    textattack.models.helpers.WordCNNForClassification.from_pretrained(
                        args.model
                    )
                )
            elif args.model.startswith("t5"):
                model = textattack.models.helpers.T5ForTextToText.from_pretrained(
                    args.model
                )
            else:
                raise ValueError(f"Unknown textattack model {args.model}")

            # Choose the approprate model wrapper (based on whether or not this is
            # a HuggingFace model).
            if isinstance(model, textattack.models.helpers.T5ForTextToText):
                model = textattack.models.wrappers.HuggingFaceModelWrapper(
                    model, model.tokenizer
                )
            else:
                model = textattack.models.wrappers.PyTorchModelWrapper(
                    model, model.tokenizer
                )
        elif args.model and os.path.exists(args.model):
            # Support loading TextAttack-trained models via just their folder path.
            # If `args.model` is a path/directory, let's assume it was a model
            # trained with textattack, and try and load it.
            if os.path.exists(os.path.join(args.model, "t5-wrapper-config.json")):
                model = textattack.models.helpers.T5ForTextToText.from_pretrained(
                    args.model
                )
                model = textattack.models.wrappers.HuggingFaceModelWrapper(
                    model, model.tokenizer
                )
            elif os.path.exists(os.path.join(args.model, "config.json")):
                with open(os.path.join(args.model, "config.json")) as f:
                    config = json.load(f)
                model_class = config["architectures"]
                if (
                    model_class == "LSTMForClassification"
                    or model_class == "WordCNNForClassification"
                ):
                    model = eval(
                        f"textattack.models.helpers.{model_class}.from_pretrained({args.model})"
                    )
                    model = textattack.models.wrappers.PyTorchModelWrapper(
                        model, model.tokenizer
                    )
                else:
                    # assume the model is from HuggingFace.
                    model = (
                        transformers.AutoModelForSequenceClassification.from_pretrained(
                            args.model
                        )
                    )
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        args.model, use_fast=True
                    )
                    model = textattack.models.wrappers.HuggingFaceModelWrapper(
                        model, tokenizer
                    )
        else:
            raise ValueError(f"Error: unsupported TextAttack model {args.model}")

        assert isinstance(
            model, textattack.models.wrappers.ModelWrapper
        ), "`model` must be of type `textattack.models.wrappers.ModelWrapper`."
        return model
