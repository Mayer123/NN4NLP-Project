import csv
import json

passages = {}
with open("../../narrativeqa/third_party/wikipedia/summaries.csv", 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    count = 0
    for line in csv_reader:
        if count == 0:
            count += 1
            print (line)
            continue
        else:
            count += 1
            passages[line[0]] = line[3]

train = []
dev = []
test = []
with open("../../narrativeqa/qaps.csv", 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    count = 0
    for line in csv_reader:
        if count == 0:
            count += 1
            print (line)
            continue
        else:
            count += 1
            sample = {}
            sample['_id'] = line[0]
            sample['context'] = passages[line[0]]
            sample['question'] = line[5]
            sample['answers'] = [line[6], line[7]]
            if line[1] == 'train':
                train.append(sample)
            elif line[1] == 'valid':
                dev.append(sample)
            elif line[1] == 'test':
                test.append(sample)
            else:
                print ('error')

print (len(train))
print (len(dev))
print (len(test))

with open("narrativeqa_summary_train.json", "w") as fout:
    json.dump(train, fout, indent=4)

with open("narrativeqa_summary_dev.json", "w") as fout:
    json.dump(dev, fout, indent=4)

with open("narrativeqa_summary_test.json", "w") as fout:
    json.dump(test, fout, indent=4)

