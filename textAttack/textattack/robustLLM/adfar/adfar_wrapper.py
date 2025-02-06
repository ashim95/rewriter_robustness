import os, sys
import argparse
import numpy as np
import torch
from torch import nn
import nltk
import random
random.seed(1337)
from copy import deepcopy
from textattack.models.wrappers import ModelWrapper
from pprint import pprint
from .modeling import BertForSequenceClassificationAdvV2
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import  BertForSequenceClassification
from .simplification import Simplifier
import re


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,simplify):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            if simplify != None:
                text_a = simplify(' '.join(text_a))
            else:
                text_a = ' '.join(text_a)
            tokens_a = tokenizer.tokenize(text_a)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, simplify,batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer,simplify)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassificationAdvV2.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data,simlifier, simplify2, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, simlifier,batch_size=batch_size)
        dataloader2 = self.dataset.transform_text(text_data, simplify2, batch_size=batch_size)

        probs_all = []
        attack_probs_all = []
        # for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for (i,((input_ids, input_mask, segment_ids), (input_ids2, input_mask2, segment_ids2))) in enumerate(zip(dataloader, dataloader2)):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            input_ids2 = input_ids2.cuda()
            input_mask2 = input_mask2.cuda()
            segment_ids2 = segment_ids2.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, input_mask,segment_ids,inference=True)
                probs1 = nn.functional.softmax(logits[0], dim=-1)
                attack_probs = logits[1]
                # print("probs1:", probs1, " probs2:", probs2, " attack_probs", attack_probs, "\n")
                probs = []

                for (ii, attack_prob) in enumerate(attack_probs):
                    if attack_prob[0] <= 0.5:
                        probs.append(probs1[ii].tolist())
                    else:

                        logits = self.model(input_ids2[ii].unsqueeze(0), input_mask2[ii].unsqueeze(0),
                                            segment_ids2 [ii].unsqueeze(0), inference=True)
                        probs1_simp = nn.functional.softmax(logits[0], dim=-1)
                        probs.append(probs1_simp.squeeze().tolist())
                probs = torch.Tensor(probs).cuda()
                attack_probs_all.append(attack_probs)
                probs_all.append(probs)

        return (torch.cat(probs_all, dim=0),torch.cat(attack_probs_all, dim=0).squeeze())


class AdfarModel(ModelWrapper):

    def __init__(self, args):
        # load everything for the main model
        self.args = args
        self.simplifier = Simplifier(threshold=3000,
                                ratio=0.3, syn_num =20,
                                most_freq_num =10)
        self.simplify_dict = {'v2': self.simplifier.simplify_v2,
                        'random_freq_v1': self.simplifier.random_freq_v1,
                        'random_freq_v2': self.simplifier.random_freq_v2}
        self.simplify = self.simplify_dict['v2']
        self.simplify2 = self.simplify_dict['random_freq_v1']
        self.nclasses = self.args.nclasses
        self.model = NLI_infer_BERT(self.args.target_model_path,
                                    nclasses=self.nclasses,
                                    max_seq_length=self.args.max_seq_length)
        self.batch_size = self.args.batch_size
        self.predictor = self.model.text_pred
        # print(self.model)

    def __clean__str__(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip() if TREC else string.strip().lower()

    def __call__(self, text_input_list):

        new_text_input_list = []
        for text in text_input_list:
            s = self.__clean__str__(text)
            new_text_input_list.append(s.split())

        probs = self.predictor(new_text_input_list, self.simplify, self.simplify2, batch_size=self.batch_size)[0]
        # logits = torch.log(probs)
        # print(text_input_list)
        # print(new_text_input_list)
        # print(probs)
        # ss
        return probs


    def get_grad(self, text_input):
        raise NotImplementedError

    def _tokenize(self, inputs):
        raise NotImplementedError
