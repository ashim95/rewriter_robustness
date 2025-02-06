from copy import deepcopy
import csv
import sys
import random
random.seed(1)
from pprint import pprint
import nltk
sent_tokenize_problem = 0.0
sent_tokenize_problem_step_2 = 0.0
def load_examples(filename):

    fp = open(filename, 'r')

    csvreader = csv.reader(fp)

    examples = []
    clean_examples = []
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        #print(row)
        if row[-1].lower() == "successful":
            ex = {}
            ex['ground'] = int(float(row[0]))
            ex['original_predicted'] = int(float(row[2]))
            #ex['original_text'] = clean_text(row[4])
            ex['original_text'] = row[4]
            ex['perturbed_predicted'] = int(float(row[5]))
            #ex['perturbed_text'] = clean_text(row[7])
            ex['perturbed_text'] = row[7]
            examples.append(ex)

    fp.close()

    print('Total number of examples ', len(examples))
    return examples

def clean_text(sentence):
    sentence = sentence.replace("[[", "")
    sentence = sentence.replace("]]", "")
    return sentence.strip()

def sentence_split(text):

    return nltk.sent_tokenize(text)

def sentence_split_adv_orig(orig, adv):

    adv_sentences = []
    orig_sentences = sentence_split(orig)

    tok_num = 0
    start = 0
    sent_num = 0
    adv_toks = adv.split(' ')
    for sent in orig_sentences:
        end = tok_num + len(sent.split(' '))
        adv_sentences.append(' '.join(adv_toks[start:end]))
        start = end

    return adv_sentences, orig_sentences

def shorten_examples(examples, extra_examples):
    # for datasets like ag_news, the input examples are very long, we can instead split into separate sentences for training

    example_set = set()
    global sent_tokenize_problem
    final_examples = []

    for ex in examples:
        adv_sentences = sentence_split(clean_text(ex['perturbed_text']))
        orig_sentences = sentence_split(clean_text(ex['original_text']))

        if len(adv_sentences) != len(orig_sentences):
            sent_tokenize_problem +=1
            # print(adv_sentences)
            # print(orig_sentences)
            adv_sentences_2, orig_sentences_2 = sentence_split_adv_orig(clean_text(ex['original_text']),
                                                                            clean_text(ex['perturbed_text']))
            # print(adv_sentences_2)
            # print(orig_sentences_2)
            if len(adv_sentences_2) != len(orig_sentences_2):
                sent_tokenize_problem_step_2 +=1
                continue
            else:
                adv_sentences = adv_sentences_2
                orig_sentences = orig_sentences_2

        for i in range(len(adv_sentences)):
            tup = (adv_sentences[i], orig_sentences[i])
            if tup in example_set:
                continue
            example_set.add(tup)
            new_ex = {}
            new_ex['perturbed_text'] = adv_sentences[i]
            new_ex['original_text'] = orig_sentences[i]
            final_examples.append(new_ex)

    final_examples_extra = []

    for ex in extra_examples:
        adv_sentences = sentence_split(clean_text(ex['perturbed_text']))
        orig_sentences = sentence_split(clean_text(ex['original_text']))

        if len(adv_sentences) != len(orig_sentences):
            sent_tokenize_problem +=1
            adv_sentences_2, orig_sentences_2 = sentence_split_adv_orig(clean_text(ex['original_text']),
                                                                            clean_text(ex['perturbed_text']))
            # print(adv_sentences_2)
            # print(orig_sentences_2)
            if len(adv_sentences_2) != len(orig_sentences_2):
                sent_tokenize_problem_step_2 +=1
                continue
            else:
                adv_sentences = adv_sentences_2
                orig_sentences = orig_sentences_2

        for i in range(len(adv_sentences)):
            tup = (adv_sentences[i], orig_sentences[i])
            if tup in example_set:
                continue
            example_set.add(tup)
            new_ex = {}
            new_ex['perturbed_text'] = adv_sentences[i]
            new_ex['original_text'] = orig_sentences[i]
            final_examples_extra.append(new_ex)

    return final_examples, final_examples_extra

def output_t5_format(filename, examples, extra_examples):

    fp1 = open(filename, 'w')
    fp1_writer = csv.writer(fp1)

    fp1_writer.writerow(["index", "adversarial", "original"])
    # also adding clean examples (so source is original sentence, target is also original sentence)

    for i in range(len(examples)):
        ex = examples[i]
        fp1_writer.writerow([str(i), clean_text(ex['perturbed_text']), clean_text(ex['original_text'])])
        fp1_writer.writerow([str(i), clean_text(ex['original_text']), clean_text(ex['original_text']) ])

    for i in range(len(extra_examples)):
        ex = extra_examples[i]
        fp1_writer.writerow([str(i), clean_text(ex['perturbed_text']), clean_text(ex['original_text'])])

    fp1.close()

    return

def check_for_perturbation(original, adversarial):

    mapping = []

    start_orig = 0
    start_adv = 0
    while True:
        start_orig = original.find("[[", start_orig)
        start_adv = adversarial.find("[[", start_adv)
        if start_adv == -1:
            return mapping
        end_orig = original.find("]]", start_orig)
        end_adv = adversarial.find("]]", start_adv)
        # adv_word = adversarial[start_adv+2:].strip().split()[0][:-2]
        # orig_word = original[start_orig+2:].strip().split()[0][:-2]
        adv_word = adversarial[start_adv+2:end_adv].strip()
        orig_word = original[start_orig+2:end_orig].strip()
        #print(adv_word)
        # print(adv_word + "\t--------\t" + orig_word)
        start_adv += 2
        start_orig += 2
        mapping.append((start_orig, start_adv, orig_word, adv_word))

    return mapping

def construct_example(example, substitutes):

    perturbed = example['original_text']

    for sub in substitutes:
        str_to_sub = "[[" + sub[2] + "]]"
        str_to_put = "[[" + sub[3] + "]]"
        # print(perturbed, str_to_sub, str_to_put)
        perturbed = perturbed.replace(str_to_sub, str_to_put)
        # print(perturbed)

    new_ex = deepcopy(example)
    new_ex['perturbed_text'] = perturbed
    return new_ex

def create_additional_examples(example, adv_orig_mapping):

    # For each example with more than one adversarial perturbation, we can generate additional
    # perturbation instances with intermediate perturbations

    num_substitutes = len(adv_orig_mapping)
    if num_substitutes < 2:
        return []

    new_examples = []
    for i in range(1, num_substitutes):
        substitutes = random.sample(adv_orig_mapping, i)
        ex = construct_example(example, substitutes)
        new_examples.append(ex)

    return new_examples



if __name__=="__main__":

    input_file = sys.argv[1]

    examples = load_examples(input_file)

    output_file = sys.argv[2]
    shorten = False
    # output_t5_format(output_file, examples)

    extra_examples = []
    i = 0
    for ex in examples:
        # print('='*30)
        # print(ex['original_text'])
        # print(ex['perturbed_text'])
        adv_orig_mapping = check_for_perturbation(ex['original_text'], ex['perturbed_text'])
        # pprint(adv_orig_mapping)
        new_examples = create_additional_examples(ex, adv_orig_mapping)
        extra_examples.extend(new_examples)
        # pprint(new_examples)
        # print('='*30)
        i +=1

    if 'ag_news' in input_file or 'ag-news' in input_file:
        if shorten:
            examples, extra_examples = shorten_examples(examples, extra_examples)
            print('Problem with sentence tokenization first type', sent_tokenize_problem)
            print('Problem with sentence tokenization after 2nd step', sent_tokenize_problem_step_2)
    output_t5_format(output_file, examples, extra_examples)
