import json

with open('/shared/data2/yuz9/MATCH/TransformerM/data/magm_train.json') as fin, open('train_texts.txt', 'w') as fou1, open('train_labels.txt', 'w') as fou2:
	for line in fin:
		data = json.loads(line)
		text = ' '.join(data['doc_token'])
		label = ' '.join(data['doc_label'])
		fou1.write(text+'\n')
		fou2.write(label+'\n')

with open('/shared/data2/yuz9/MATCH/TransformerM/data/magm_dev.json') as fin, open('train_texts.txt', 'a') as fou1, open('train_labels.txt', 'a') as fou2:
	for line in fin:
		data = json.loads(line)
		text = ' '.join(data['doc_token'])
		label = ' '.join(data['doc_label'])
		fou1.write(text+'\n')
		fou2.write(label+'\n')

with open('/shared/data2/yuz9/MATCH/TransformerM/data/magm_test.json') as fin, open('test_texts.txt', 'w') as fou1, open('test_labels.txt', 'w') as fou2:
	for line in fin:
		data = json.loads(line)
		text = ' '.join(data['doc_token'])
		label = ' '.join(data['doc_label'])
		fou1.write(text+'\n')
		fou2.write(label+'\n')