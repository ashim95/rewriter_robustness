import os, sys
from argparse import Namespace


def get_default_cls_model_args():
    args = Namespace()

    args.model_revision = "main"
    args.use_auth_token = False
    args.use_fast_tokenizer = True
    args.cache_dir = None
    args.config_name = None
    args.tokenizer_name = None
    args.ignore_mismatched_sizes = False
    return args

def get_default_interceptor_args():
    args = Namespace()

    args.model_revision = "main"
    args.use_auth_token = False
    args.use_fast_tokenizer = True
    args.cache_dir = None
    args.config_name = None
    args.tokenizer_name = None
    args.lang = None
    args.source_prefix = ""
    args.max_target_length = None

    return args


def cls_preprocess_function(examples, tokenizer, max_length, padding):

    # TODO: Handle for TextPair Tasks like NLI
    # if not isinstance(len(examples[0]), tuple):
    #     texts = [(e, ) for e in examples]
    # print(texts)
    result = tokenizer(examples, padding=padding, max_length=max_length, truncation=True)
    return result
