import csv
import sys

def calculate_metrics(filename):

    fp = open(filename, 'r')

    csvreader = csv.reader(fp)

    total = 0.0
    correct = 0.0
    num_queries_success = []
    num_queries_all = []
    attack_success = 0.0
    clean_incorrect_indices = []

    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        #print(row)
        total +=1
        ground = int(float(row[0]))
        original_predicted = int(float(row[2]))
        perturbed_predicted = int(float(row[5]))
        queries = int(float(row[1]))
        result = row[-1]
        num_queries_all.append(queries)
        if ground == original_predicted:
            correct +=1
            if perturbed_predicted != original_predicted:
                attack_success +=1
                num_queries_success.append(queries)
        else:
            clean_incorrect_indices.append(i)



    fp.close()

    asr = attack_success/correct
    print('Total number of examples ', total)

    print('Printing report \n' + '-'*20)

    print('Original Accuracy: ', round(100*correct/total, 2))
    print('Attack Success Rate: {}% '.format(100*attack_success/correct))
    print('Adversarial Accuracy: {}%'.format(round(100*(1 - asr), 2)))
    print('Average Number of Queries (all): ', sum(num_queries_all)/len(num_queries_all))
    print('Average Number of Queries (success): ', sum(num_queries_success)/len(num_queries_success))
    print('-'*20)

    print('Indices of the clean examples where the model predicted incorrect originally ')
    print(clean_incorrect_indices)

def clean_text(sentence):
    sentence = sentence.replace("[[", "")
    sentence = sentence.replace("]]", "")
    return sentence.strip()

if __name__=="__main__":

    textattack_file = sys.argv[1]

    calculate_metrics(textattack_file)
