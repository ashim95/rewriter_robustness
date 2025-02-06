import torch
import numpy as np
from sklearn.metrics import f1_score
# import OpenAttack as oa
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
# from dataset import *
from .model import BertClassifierDARTS
from textattack.models.wrappers import ModelWrapper
from .utils import prepare_single_bert, get_preds
from transformers import AutoTokenizer

class ShieldModel(ModelWrapper):

    def __init__(self, args):

        self.args = args
        self.model_path = args.model_path
        self.max_seq_len = args.max_seq_len
        self.device = args.device

        self.model = self.load_model(args)
        self.tokenizer = self.load_tokenizer(args)

    def load_model(self, args):

        if args.base_classifier:
            model = BertClassifierDARTS(model_type=args.model_type,
                                        freeze_bert=False,
                                        output_dim=args.nclasses,
                                        ensemble=0,
                                        device=args.device)
        else:
            model = BertClassifierDARTS(model_type=args.model_type,
                                        freeze_bert=True,
                                        is_training=False,
                                        inference=True,
                                        output_dim=args.nclasses,
                                        ensemble=1,
                                        N=5,
                                        temperature=args.temperature,
                                        gumbel=1,
                                        scaler=1,
                                        darts=True,
                                        device=args.device)
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(args.device)
        model.eval()
        return model

    def load_tokenizer(self, args):

        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        return tokenizer

    def __call__(self, text_input_list):

        data_iter = prepare_single_bert(text_input_list,
                                        tokenizer=self.tokenizer,
                                        batch_size=1,
                                        max_len=self.max_seq_len,
                                        device=self.device)
        preds = get_preds(self.model, data_iter, return_pt_tensor=True)
        # print(preds)
        return preds

    def get_grad(self, text_input):
        raise NotImplementedError

    def _tokenize(self, inputs):
        raise NotImplementedError
