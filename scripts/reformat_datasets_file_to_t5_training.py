import csv
import sys


def load_examples(filename):

    fp = open(filename, 'r')

    csvreader = csv.reader(fp)

    examples = []
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        #print(row)
        ex = {}
        ex['idx'] = int(row[0])
        ex['label'] = int(row[-1])
        ex['sentence1'] = row[1].strip()
        examples.append(ex)

    fp.close()

    print('Total number of examples ', len(examples))
    return examples

def clean_text(sentence):
    sentence = sentence.replace("[[", "")
    sentence = sentence.replace("]]", "")
    return sentence.strip()

def output_t5_format(filename, examples):

    fp1 = open(filename, 'w')
    fp1_writer = csv.writer(fp1)

    fp1_writer.writerow(["index", "adversarial", "original"])

    for i in range(len(examples)):
        ex = examples[i]
        fp1_writer.writerow([str(i), ex['sentence1'], ex['sentence1']])

    fp1.close()

    return

if __name__=="__main__":

    input_file = sys.argv[1]

    examples = load_examples(input_file)

    output_file = sys.argv[2]

    output_t5_format(output_file, examples)
