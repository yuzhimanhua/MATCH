import json
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG', choices=['MAG', 'MeSH'])

args = parser.parse_args()
folder = args.dataset

with open(f'{folder}/meta_dict.json') as fin:
	meta_dict = json.load(fin)
	meta_set = meta_dict['metadata']

data_files = ['train', 'dev', 'test']
for data_file in data_files:
	if data_file == 'train':
		output_file = 'train'
		mode = 'w'
	elif data_file == 'dev':
		output_file = 'train'
		mode = 'a'
	else:
		output_file = 'test'
		mode = 'w'
	
	with open(f'{folder}/{data_file}.json') as fin, open(f'{folder}/{output_file}_texts.txt', mode) as fou1, open(f'{folder}/{output_file}_labels.txt', mode) as fou2:
		for line in fin:
			data = json.loads(line)

			metadata = []
			for meta in meta_set:
				if type(data[meta]) is not list:
					metadata.append(meta.upper()+'_'+data[meta].replace(' ', '_'))
				else:
					for x in data[meta]:
						metadata.append(meta.upper()+'_'+x.replace(' ', '_'))
			
			text = ' '.join(metadata) + ' ' + data['text']
			label = ' '.join(data['label'])
			fou1.write(text+'\n')
			fou2.write(label+'\n')
