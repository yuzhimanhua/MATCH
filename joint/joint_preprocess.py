import json
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='MAG', choices=['MAG', 'MeSH'])

args = parser.parse_args()
folder = f'../{args.dataset}/'

thrs = 5
left = set()
right = set()

with open(folder+'meta_dict.json') as fin:
	meta_dict = json.load(fin)
	meta_set = meta_dict['metadata']

node2cnt = defaultdict(int)
with open(folder+'train.json') as fin:
	for idx, line in enumerate(fin):
		if idx % 10000 == 0:
			print(idx)
		data = json.loads(line)
		
		for W in data['text'].split():
			node2cnt[W] += 1
		
		for meta in meta_set:
			if type(data[meta]) != list:
				data[meta] = [data[meta]]
			for x in data[meta]:
				M = meta.upper()+'_'+x.replace(' ', '_')
				node2cnt[M] += 1

with open(folder+'train.json') as fin, open('network.dat', 'w') as fout:
	for idx, line in enumerate(fin):
		if idx % 10000 == 0:
			print(idx)
		data = json.loads(line)

		P = 'PAPER_'+data['paper']
		left.add(P)
		
		# P-L		
		for L0 in data['label']:
			L = 'LABEL_' + L0
			fout.write(P+' '+L+' 0 1 \n')
			right.add(L)

		# P-M
		for meta in meta_set:
			if type(data[meta]) != list:
				data[meta] = [data[meta]]
			for x in data[meta]:
				M = meta.upper()+'_'+x.replace(' ', '_')
				if node2cnt[M] >= thrs:
					fout.write(P+' '+M+' 1 1 \n')
					right.add(M)

		# P-W
		words = data['text'].split()
		for W in words:
			if node2cnt[W] >= thrs:
				fout.write(P+' '+W+' 2 1 \n')
				right.add(W)

		# Wc-W
		for i in range(len(words)):
			Wi = words[i]
			if node2cnt[Wi] < thrs:
				continue
			for j in range(i-5, i+6):
				if j < 0 or j >= len(words) or j == i:
					continue
				Wj = words[j]
				if node2cnt[Wj] < thrs:
					continue
				fout.write(Wj+' '+Wi+' 3 1 \n')
				left.add(Wj)
			
with open('left.dat', 'w') as fou1, open('right.dat', 'w') as fou2:
	for x in left:
		fou1.write(x+'\n')
	for x in right:
		fou2.write(x+'\n')