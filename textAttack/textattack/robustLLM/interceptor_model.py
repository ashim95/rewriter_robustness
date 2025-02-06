import os, sys
import argparse
import numpy as np
import torch
from torch import nn
import nltk
import random
random.seed(1337)
from copy import deepcopy
from . import cls_preprocess_function
from torch.utils.data import DataLoader
from textattack.models.wrappers import ModelWrapper
from pprint import pprint

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from datasets import Dataset

class FullModel(ModelWrapper):

    """
    The model wrapper to integrate with textattack.
    For an input, first pass it through the LLM/T5 model, get argmax string outputs.
    Then pass it through the classification model
    """

    def __init__(self, cls_args=None, interceptor_args=None):
        self.cls_args = cls_args
        self.interceptor_args = interceptor_args
        self.cls_config, self.cls_tokenizer, self.cls_model = self.load_cls(cls_args)
        self.interceptor_config, self.interceptor_tokenizer, self.interceptor_model = self.load_interceptor(interceptor_args)

        self.interceptor_data_collator = DataCollatorForSeq2Seq(
        self.interceptor_tokenizer,
        model=self.interceptor_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        )

        self.generate_sentence_by_sentence = interceptor_args.generate_sentence_by_sentence

        self.generation_method = interceptor_args.generation_method

    def load_interceptor(self, args):
        print(args.model_name_or_path)
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )

        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        model.to(args.device)

        return config, tokenizer, model

    def load_cls(self, args):

        print(args)
        config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )

        model.to(args.device)
        return config, tokenizer, model

    def to(self, device):
        self.interceptor_model.to(device)
        self.cls_model.to(device)

    def postprocess_denoising(self, generated_delta, text_list):

        big_splitter = "Z>"
        replacement_splitter = "Y>"

        # print(text_list)
        new_text_list = []
        for i in range(len(text_list)):
            text = text_list[i]
            delta = generated_delta[i]
            changes = delta.split('Z>')
            print(changes)
            for ch in changes:
                if ch.strip() == '':
                    continue
                splits = ch.split('X>Y')
                if len(splits) != 2:
                    continue
                to_replace_word = ch.split('X>Y')[0][2:].strip()
                if len(to_replace_word) < 1:
                    continue
                replaced_word = ch.split('X>Y')[1][1:-2]
                if to_replace_word in text:
                    text = text.replace(to_replace_word, replaced_word)

            new_text_list.append(text)

        # print(new_text_list)
        return new_text_list


    def postprocess_denoising_2(self, generated_delta, text_list):

        big_splitter = "##"
        replacement_splitter = ">"

        # print(text_list)
        # print(generated_delta)
        new_text_list = []
        for i in range(len(text_list)):
            text = text_list[i]
            delta = generated_delta[i]
            changes = delta.split(big_splitter)
            # print(changes)
            for ch in changes:
                if ch.strip() == '':
                    continue
                splits = ch.split(replacement_splitter)
                if len(splits) != 2:
                    continue
                to_replace_word = ch.split(replacement_splitter)[0].strip()
                if len(to_replace_word) < 1:
                    continue
                replaced_word = ch.split(replacement_splitter)[1].strip()
                if to_replace_word in text:
                    text = text.replace(to_replace_word, replaced_word)

            new_text_list.append(text)

        # print(new_text_list)
        return new_text_list

    def __call__(self, text_input_list, batch_size=32):

        # model_device = next(self.model.parameters()).device
        model_device = self.cls_model.device
        # Run through the interceptor model
        # in this case the interceptor is the random model

        # print(self.interceptor_args.percent_random_defense)
        # pprint(text_input_list)
        # new_text_input_list = self.batched_random_synonym_replacement(text_input_list,
        #                             self.interceptor_args.percent_random_defense)
        new_text_input_list = self.get_t5_prediction(text_input_list, generate_sentence_by_sentence=self.generate_sentence_by_sentence)

        if self.generation_method == 'denoise':
            # new_text_input_list = self.postprocess_denoising(new_text_input_list, text_input_list)
            new_text_input_list = self.postprocess_denoising_2(new_text_input_list, text_input_list)

        # new_text_input_list = text_input_list
        # Now run the model through the classifier module
        # Run the tokenizer, get the ids
        # Run forward evaluation on the cls model
        # pprint(new_text_input_list)
        return self.make_cls_prediction(new_text_input_list)

    def get_grad(self, text_input):
        raise NotImplementedError

    def _tokenize(self, inputs):
        raise NotImplementedError

    def sentence_split(self, text):

        return nltk.sent_tokenize(text)

    def prepare_interceptor_text(self, texts):
        inputs = [self.interceptor_args.source_prefix + inp for inp in texts]
        model_inputs = self.interceptor_tokenizer(inputs, max_length=self.interceptor_args.max_target_length, padding="max_length", truncation=True)

        return model_inputs

    def postprocess_interceptor_output(self, preds):
        preds = [pred.strip() for pred in preds]
        return preds

    def get_t5_prediction(self, text_input, generate_sentence_by_sentence=False):
        # first preprocess text
        # print(text_input)
        if generate_sentence_by_sentence: # if we want to generate from t5 sentence by sentence
            #pprint(text_input)
            new_text_input = []
            for text in text_input:
                text_sents = self.sentence_split(text)
                out_sents = self.get_t5_prediction(text_sents)
                new_text_input.append(' '.join(out_sents))
            #print(generate_sentence_by_sentence)
            #print('--'*20)
            #pprint(new_text_input)

            return new_text_input

        model_inputs = self.prepare_interceptor_text(text_input)
        # print(model_inputs)
        eval_dataset = Dataset.from_dict(model_inputs)

        # then get prediction from t5 model
        eval_dataloader = DataLoader(eval_dataset, collate_fn=self.interceptor_data_collator, batch_size=self.cls_args.batch_size)

        gen_kwargs = {
            "max_length": self.interceptor_args.max_target_length,
            "num_beams": 5,
        }
        outputs_texts = []

        # postprocess prediction and return
        self.interceptor_model.eval()
        for step, batch_cpu in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k:v.to(self.cls_args.device) for k,v in batch_cpu.items()}
                generated_tokens = self.interceptor_model.generate(
                                        batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        **gen_kwargs,
                                            )
                generated_tokens = generated_tokens.cpu().numpy()
                # print(generated_tokens)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.interceptor_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                final_outs = self.postprocess_interceptor_output(decoded_preds)
                # sss
                outputs_texts.extend(final_outs)

        return outputs_texts


    def make_cls_prediction(self, text_list):

        processed_list = cls_preprocess_function(text_list, self.cls_tokenizer, self.cls_args.cls_max_seq_length, "max_length")

        eval_dataset = Dataset.from_dict(processed_list)

        # print(type(ad))
        # print(ad)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=self.cls_args.batch_size)

        self.cls_model.eval()

        all_logits = None
        all_predictions = None
        for step, batch_cpu in enumerate(eval_dataloader):
            # print(batch_cpu)
            batch = {k:v.to(self.interceptor_args.device) for k,v in batch_cpu.items()}
            with torch.no_grad():
                outputs = self.cls_model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            if all_logits is None:
                all_logits = outputs.logits
                all_predictions = predictions
            else:
                all_logits = torch.cat([all_logits, outputs.logits], 0)
                all_predictions = torch.cat([all_predictions, predictions])
        # print(all_predictions.shape)
        # print(all_logits.shape)
        return all_logits

class FullRandomModel(ModelWrapper):

    """
    The model wrapper to integrate with textattack.
    For an input, first pass it through the LLM/T5 model, get argmax string outputs.
    Then pass it through the classification model
    """

    def __init__(self, cls_args=None, interceptor_args=None):
        self.cls_args = cls_args
        self.interceptor_args = interceptor_args
        self.cls_config, self.cls_tokenizer, self.cls_model = self.load_cls(cls_args)
        self.interceptor_model = self.load_interceptor(interceptor_args)


    def load_interceptor(self, args):

        from transformers import pipeline
        fillmask_bert = pipeline('fill-mask', model="bert-base-uncased", top_k=args.topk_random_defense, device=0)

        return fillmask_bert

    def load_cls(self, args):

        print(args)
        config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )

        model.to(args.device)

        return config, tokenizer, model

    # def to(self, device):
    #     self.interceptor_model.to(device)
    #     self.cls_model.to(device)

    def __call__(self, text_input_list, batch_size=32):

        # model_device = next(self.model.parameters()).device
        model_device = self.cls_model.device
        # Run through the interceptor model
        # in this case the interceptor is the random model

        # print(self.interceptor_args.percent_random_defense)
        # pprint(text_input_list)
        new_text_input_list = self.batched_random_synonym_replacement(text_input_list,
                                    self.interceptor_args.percent_random_defense)

        # new_text_input_list = text_input_list
        # Now run the model through the classifier module
        # Run the tokenizer, get the ids
        # Run forward evaluation on the cls model
        # pprint(new_text_input_list)
        return self.make_cls_prediction(new_text_input_list)

    def get_grad(self, text_input):
        raise NotImplementedError

    def _tokenize(self, inputs):
        raise NotImplementedError

    def fill_masked_word(self, toks, position):

        new_toks = deepcopy(toks)
        new_toks[position] = '[MASK]'

        s = ' '.join(new_toks)

        words = self.interceptor_model(s)
        return words

    def random_synonym_replacement(self, toks, num):

        possible_indices = list(range(len(toks)))


        random.shuffle(possible_indices)
        # global skipped
        try:
            selected_indices = random.sample(possible_indices, num)
        except:
            print('Exception occurred , skipping example')
            # skipped +=1
            return [' '.join(toks)] * 5

        replaced_list = [] # list of lists

        for position in selected_indices:
            words = self.fill_masked_word(toks, position)
            if len(words) > 0:
                l = [w['token_str'] for w in words]
            replaced_list.append((position, l))

        new_sentences = []

        for i in range(len(l)):
            new_toks = deepcopy(toks)
            for replaced in replaced_list:
                new_toks[replaced[0]] = replaced[1][i]
            new_sentences.append(' '.join(new_toks))

        return new_sentences

    def batched_random_synonym_replacement(self, text_input_list, percent):

        randomized_inputs = []
        for text in text_input_list:
            toks = nltk.word_tokenize(text)
            num_substitutes = max(1, round(percent*len(toks)))
            # print(num_substitutes)
            new_sentences = self.random_synonym_replacement(toks, num_substitutes)
            # Randomly select one of the transformed sentences
            new_sent = random.choice(new_sentences)
            randomized_inputs.append(new_sent)

        return randomized_inputs

    def make_cls_prediction(self, text_list):

        processed_list = cls_preprocess_function(text_list, self.cls_tokenizer, self.cls_args.cls_max_seq_length, "max_length")

        eval_dataset = Dataset.from_dict(processed_list)

        # print(type(ad))
        # print(ad)

        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=self.cls_args.batch_size)

        self.cls_model.eval()


        all_logits = None
        all_predictions = None
        for step, batch_cpu in enumerate(eval_dataloader):
            # print(batch_cpu)
            batch = {k:v.to(self.cls_args.device) for k,v in batch_cpu.items()}
            with torch.no_grad():
                outputs = self.cls_model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            if all_logits is None:
                all_logits = outputs.logits
                all_predictions = predictions
            else:
                all_logits = torch.cat([all_logits, outputs.logits], 0)
                all_predictions = torch.cat([all_predictions, predictions])
        # print(all_predictions.shape)
        # print(all_logits.shape)
        return all_logits
