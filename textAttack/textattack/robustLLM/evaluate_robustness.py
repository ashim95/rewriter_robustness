import os, sys
import argparse
from utils import get_default_cls_model_args, get_default_interceptor_args
from interceptor_model import FullModel, FullRandomModel
import torch
import pandas as pd

def load_texts(filename, column=None):

    df = pd.read_csv(filename)

    if column is None:
        return df
    return list(df[column])




if __name__=='__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--cls_model", type=str, default=None, required=True)
    parser.add_argument("--interceptor_model", type=str, default=None, required=True)
    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--variant", type=str, default="random", required=True)
    # parser.add_argument("--method", type=str, default=None)

    parser.add_argument("--source_prefix", type=str, default="correct the given sentence: ")
    parser.add_argument("--percent_random_defense", type=float, default=0.1)
    parser.add_argument("--topk_random_defense", type=float, default=5)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--num_labels", type=int, default=2)
    # parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--cls_max_seq_length", type=int, default=128)


    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cls_model_args = get_default_cls_model_args()
    interceptor_args = get_default_interceptor_args()

    if not args.interceptor_model.lower() == 'random':
        interceptor_args.model_name_or_path = args.interceptor_model
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


    full_model = FullRandomModel(interceptor_args=interceptor_args, cls_args=cls_model_args)

    text_list = [
        "leather new secretions from the parental unity",
        "contains no waite , only labored laughs",
        "coolest movie",
        "lend some dignity to a dumb story",
    ]

    text_list = load_texts(args.input_file, column="sentence1")
    print('Number of texts loaded ', len(text_list))
    print(text_list[:5])


    full_model.make_cls_prediction(text_list)