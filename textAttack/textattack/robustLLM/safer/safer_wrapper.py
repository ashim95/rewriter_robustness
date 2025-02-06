from xml.sax.handler import all_properties
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from .dataset_utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

from .dataset_utils import InputExample
from .data_util import WordSubstitude, set_seed


import pickle
import string

from transformers import (BertConfig,
                          BertForSequenceClassification,
                          BertTokenizer,
                          XLMConfig,
                          XLMForSequenceClassification,
                          XLMTokenizer,
                          XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer
                          )


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


class SaferModel(ModelWrapper):

    def __init__(self, args):

        self.args = args
        self.device = args.device

        self.model, self.config, self.tokenizer = self.load_model(args)

        # set random seed
        set_seed(args)

        # load word embeddings (counter-fitting)
        # self.word_emb = self.load_word_emb(args.word_emb_file)
        # load perturbation table
        self.perturb_pca = self.load_perturb_pca(args.perturb_pca_path)
        self.random_smoother = WordSubstitude(self.perturb_pca)


    def load_word_emb(path):
        word_embd = {}
        # embd_file = os.path.join(embd_path, 'counter-fitted-vectors.txt')
        with open(path, "r") as f:
            tem = f.readlines()
            for line in tem:
                line = line.strip()
                line = line.split(' ')
                word = line[0]
                vec = line[1:len(line)]
                vec = [float(i) for i in vec]
                vec = np.asarray(vec)
                word_embd[word] = vec

        # Name = data_path + '/word_embd.pkl'
        # output = open(Name, 'wb')
        # pickle.dump(word_embd, output)
        # output.close()

        return word_embd

    def load_perturb_pca(self, perturb_pca_path):

        # rewrite this function
        # pkl_file = open(args.temp_data_dir + args.task_name + '_perturbation_constraint_pca' + str(args.similarity_threshold) + '_' + str(args.perturbation_constraint) + '.pkl', 'rb')
        pkl_file = open(perturb_pca_path, 'rb')
        perturb_pca = pickle.load(pkl_file)
        pkl_file.close()

        # shorten the perturbation set to desired length constraint
        for key in perturb_pca.keys():
            if len(perturb_pca[key]['set'])>self.args.perturbation_constraint:

                tem_neighbor_count = 0
                tem_neighbor_list = []
                for tem_neighbor in perturb_pca[key]['set']:
                    tem_neighbor_list.append(tem_neighbor)
                    tem_neighbor_count += 1
                    if tem_neighbor_count >= self.args.perturbation_constraint:
                        break
                perturb_pca[key]['set'] = tem_neighbor_list
                perturb_pca[key]['isdivide'] = 1

        return perturb_pca

    def load_model(self, args):
        args.config_name = None
        args.tokenizer_name = None
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels, finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        return model, config, tokenizer

    def __call__(self, text_input_list):

        # first for each input, smooth and generate smoothed sentences

        all_probs = None

        for text in text_input_list:
            samples = []
            samples.append(text)
            for _ in range(self.args.num_random_samples):
                sample = str(self.random_smoother.get_perturbed_batch(np.array([[text]]))[0][0])
                samples.append(sample)

            # Run evaluation for all generated samples
            probs, preds = self.make_cls_prediction(samples, get_preds=True) # shape = len(samples), num_classes
            # print(probs.shape)
            probs = torch.mean(probs, dim=0)
            # print(probs.shape)
            # print(probs)
            # average for the probs
            # majority for the label
            if all_probs is None:
                all_probs = torch.unsqueeze(probs, 0)
            else:
                all_probs = torch.cat([all_probs, torch.unsqueeze(probs, 0)], 0)
        return all_probs

    def convert_to_dataset(self, examples, label_list):

        features = convert_examples_to_features(examples, label_list, self.args.max_seq_length, self.tokenizer, "classification",
            cls_token_at_end=bool(self.args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 1,
            pad_on_left=bool(self.args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)

            # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def make_cls_prediction(self, text_list, get_preds=False):

        examples = []

        processor = processors[self.args.task_name]()

        label_list = processor.get_labels()

        for i, text in enumerate(text_list):
            text_a = text.strip()
            guid = "%s-%d" % ('test', i); i+=1
            text_b = None # TODO: for sentence pair tasks
            label = label_list[0] # just assign any label since we just want to get the output prediction,
                                  # not calcuation of any metric
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        eval_dataset = self.convert_to_dataset(examples, label_list)

        # print(type(ad))
        # print(ad)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        self.model.eval()


        all_logits = None
        all_predictions = None
        for step, batch_cpu in enumerate(eval_dataloader):
            # print(batch_cpu)
            batch = tuple(t.to(self.args.device) for t in batch_cpu)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if self.args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                'labels':         batch[3]}
                outputs = self.model(**inputs)

                tmp_eval_loss, logits = outputs[:2]

            predictions = logits.argmax(dim=-1)
            if all_logits is None:
                all_logits = logits
                all_predictions = predictions
            else:
                all_logits = torch.cat([all_logits, logits], 0)
                all_predictions = torch.cat([all_predictions, predictions])
        # print(all_predictions.shape)
        # print(all_logits.shape)
        all_logits = torch.nn.functional.softmax(all_logits, dim=1)

        # get_average for logits
        # get majority for label

        if get_preds:
            return all_logits, all_predictions
        return all_logits



    def get_grad(self, text_input):
        raise NotImplementedError

    def _tokenize(self, inputs):
        raise NotImplementedError
