import os, sys, nltk
from transformers import pipeline
from copy import deepcopy
import csv

import random
random.seed(1)

skipped = 0.0

fillmask_bert = pipeline('fill-mask', model="bert-base-uncased")


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
			'ours', 'ourselves', 'you', 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his',
			'himself', 'she', 'her', 'hers', 'herself',
			'it', 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who',
			'whom', 'this', 'that', 'these', 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at',
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again',
			'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each',
			'few', 'more', 'most', 'other', 'some', 'such', 'no',
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now', '']

def fill_masked_word(toks, position):

    new_toks = deepcopy(toks)
    new_toks[position] = '[MASK]'

    s = ' '.join(new_toks)

    words = fillmask_bert(s)
    return words


def load_examples(filename):

    examples = {}
    fp = open(filename, 'r')

    csvreader = csv.reader(fp)

    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        index = int(row[0])
        examples[index] = (row[1], row[-1])

    return examples

def synonym_substitution(toks, num):

    possible_indices = [i for i in range(len(toks)) if toks[i].lower() not in stop_words]

    random.shuffle(possible_indices)
    global skipped
    try:
        selected_indices = random.sample(possible_indices, num)
    except:
        print('Exception occurred , skipping example')
        skipped +=1
        return [' '.join(toks)] * 5

    replaced_list = [] # list of lists

    for position in selected_indices:
        words = fill_masked_word(toks, position)
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

def generate_random_examples(example, ratio=0.2):

    toks = example.lower().split()
    num_substitutes = max(1, round(ratio*len(toks)))

    new_sentences = synonym_substitution(toks, num_substitutes)
    return new_sentences

def output_hf_format(filename1, examples, write=1):

    fp1 = open(filename1, 'w')
    fp1_writer = csv.writer(fp1)

    fp1_writer.writerow(["file_sentence_index", "sentence1", "random_sample_num", "label"])

    for key, val in examples.items():
        for n in range(write):
            ex = examples[key][n]
            label = ex[-1]
            index = str(key)
            random_sample_num = str(ex[1])
            text = ex[2].strip()
            fp1_writer.writerow([index, text, random_sample_num, label])

    fp1.close()

if __name__=="__main__":

    input_file = sys.argv[1]

    output_file = sys.argv[2]

    ratio = float(sys.argv[3])

    examples = load_examples(input_file)

    new_examples = {}

    done = 0
    for index, ex in examples.items():
        new_sentences = generate_random_examples(ex[0], ratio=ratio)
        for i, sent in enumerate(new_sentences):
            if index not in new_examples:
                new_examples[index] = []
            new_examples[index].append((index, i, sent, ex[1]))
            #new_examples.append((index, i, sent, ex[1]))
        done +=1
        if done % 100 == 0:
            print('Done:{}/{}'.format(done, len(examples)))

    output_hf_format(output_file, new_examples, write=len(new_sentences))

    print('Examples that should be skipped ', skipped)
