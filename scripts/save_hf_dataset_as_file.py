import os, sys
from datasets import load_dataset
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default="glue")
parser.add_argument("--dataset_name", default=None, required=False)
parser.add_argument("--split", default=None, required=True)
parser.add_argument("--output_file", default=None, required=True)
parser.add_argument("--is_pair", action='store_true', required=False)

args = parser.parse_args()


def get_dataset(benchmark, name):
    data = load_dataset(benchmark, name)
    return data


def write_dataset(dataset, split, output_file):

    fp = open(output_file, 'w')
    csvwriter = csv.writer(fp)

    csvwriter.writerow(["index", "sentence1", "label"])

    for i, ex in enumerate(dataset[split]):
        if 'idx' not in ex:
            csvwriter.writerow([i, ex['text'], ex['label']])
        else:
            csvwriter.writerow([ex['idx'], ex['sentence'], ex['label']])

    fp.close()

def write_dataset_pair(dataset, split, output_file, is_mnli=True):

    fp = open(output_file, 'w')
    csvwriter = csv.writer(fp)

    csvwriter.writerow(["index", "sentence1", "sentence2", "label"])
    if is_mnli:
        sentence1_key = 'premise'
        sentence2_key = 'hypothesis'

    for i, ex in enumerate(dataset[split]):
        if len(ex[sentence1_key].strip()) <2 or len(ex[sentence2_key].strip()) <2:
            print('Skipping example index ', ex['idx'])
            print('sentence1: {}, sentence2:{}'.format(ex[sentence1_key], ex[sentence2_key]))
            continue
        if 'idx' not in ex:
            csvwriter.writerow([i, ex[sentence1_key], ex[sentence2_key],  ex['label']])
        else:
            csvwriter.writerow([ex['idx'], ex[sentence1_key], ex[sentence2_key],  ex['label']])

    fp.close()

dataset = get_dataset(args.benchmark, args.dataset_name)
if args.is_pair:
    write_dataset_pair(dataset, args.split, args.output_file)
else:
    write_dataset(dataset, args.split, args.output_file)
