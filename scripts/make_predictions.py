import os, sys, csv

def read_preds(filename):

    preds = []
    fp = open(filename, 'r')
    for line in fp:
        if 'index' in line:
            continue
        preds.append(line.strip().split('\t')[1])

    return preds


def load_examples(filename, preds):

    examples = {}

    fp = open(filename, 'r')
    csvreader = csv.reader(fp)
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        pred = preds[i-1]
        orig_label = row[-1]
        example_index = int(row[0])
        perturbation_index = int(row[2])
        sentence = row[1]
        if example_index not in examples:
            examples[example_index] = []
        examples[example_index].append((perturbation_index, sentence, pred, orig_label))

    return examples


def make_first_prediction(examples):
    return examples[0][2]

def most_common(lst):
    return max(lst, key=lst.count)


def make_majority_prediction(examples):
    preds = [ex[2] for ex in examples]

    return most_common(preds)


def get_accuracy(examples):

    correct_first = 0.0
    correct_majority = 0.0
    total = len(examples)

    for key, val in examples.items():
        gold = val[0][-1]
        first = make_first_prediction(val)
        majority = make_majority_prediction(val)

        if first == gold:
            correct_first +=1
        if majority == gold:
            correct_majority +=1

    print('Accuracy with selecting the label of first perturbation {}'.format(correct_first/total))
    print('Accuracy with selecting the majority of all perturbations {}'.format(correct_majority/total))

    return correct_first/total, correct_majority/total


if __name__=="__main__":

    examples_file = sys.argv[1]
    preds_file = sys.argv[2]

    preds = read_preds(preds_file)
    examples = load_examples(examples_file, preds)
    first, majority = get_accuracy(examples)

    if len(sys.argv) > 3:
        ratio = sys.argv[3]
        save_results_file = sys.argv[4]
        fp = open(save_results_file, 'a')
        fp.write(str(ratio) + ',' + str(first) + ',' + str(majority) + '\n')
        fp.close()
