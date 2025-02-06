import csv
import sys

def load_examples(filename):

    fp = open(filename, 'r')

    csvreader = csv.reader(fp)

    examples = []
    j = 0
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        #print(row)
        if row[-1].lower() == "successful":
            ex = {}
            ex['ground'] = int(float(row[0]))
            ex['original_predicted'] = int(float(row[2]))
            ex['original_text'] = row[4]
            ex['perturbed_predicted'] = int(float(row[5]))
            ex['perturbed_text'] = row[7]
            ex['idx'] = j
            j +=1
            examples.append(ex)

    fp.close()

    print('Total number of examples ', len(examples))
    return examples

def clean_text(sentence):
    sentence = sentence.replace("[[", "")
    sentence = sentence.replace("]]", "")
    return sentence.strip()

def load_perturbed_file_and_preds(filename, preds_filename):

    fp1 = open(filename, 'r')
    fp2 = open(preds_filename, 'r')

    csv1 = csv.reader(fp1)
    csv2 = csv.reader(fp2, delimiter='\t')

    examples = []
    csv2 = list(csv2)
    for i, row in enumerate(csv1):
        if i ==0:
            continue
        idx = int(row[0])
        sentence = row[1]
        pred = int(csv2[i][-1])
        examples.append((idx, sentence, pred))
    return examples

def output_hf_format(filename1, filename2, examples):

    fp1 = open(filename1, 'w')
    fp2 = open(filename2, 'w')
    fp1_writer = csv.writer(fp1)
    fp2_writer = csv.writer(fp2)

    fp1_writer.writerow(["index", "sentence1", "label"])
    fp2_writer.writerow(["index", "sentence1",  "label"])

    for i in range(len(examples)):
        ex = examples[i]
        fp1_writer.writerow([str(i), ex['original_text'], ex['ground']])
        fp2_writer.writerow([str(i), ex['perturbed_text'], ex['ground']])

    fp1.close()
    fp2.close()

def check_for_perturbation(original, adversarial, perturbed):

    start = 0
    while True:
        start = adversarial.find("[[", start)
        if start == -1:
            return True
        adv_word = adversarial[start+2:].strip().split()[0][:-2]
        #print(adv_word)
        if adv_word not in perturbed:
            print(adv_word + "\t--------\t" + perturbed)
            return False
        start += 2

    return True



if __name__=="__main__":

    input_file = sys.argv[1]

    examples = load_examples(input_file)

    perturbed = load_perturbed_file_and_preds(sys.argv[2], sys.argv[3])

    correct = 0.0
    adv_replacements = 0.0

    for i in range(len(perturbed)):
        idx = perturbed[i][0]
        if examples[idx]['ground'] == perturbed[i][-1]:
            correct +=1
            if not check_for_perturbation(examples[idx]['original_text'], examples[idx]['perturbed_text'], perturbed[i][1]):
                adv_replacements +=1

    print('Crude Accuracy ', correct/len(perturbed))
    print('Adversarial Replacements ', adv_replacements/correct)


