import json
from collections import defaultdict

folder = '/shared/data2/yuz9/MATCH/MeSH_data/'
tot = 898546
thrs = 5

left = set()
right = set()

node2cnt = defaultdict(int)
venue2id = {}
with open(folder+'MeSH.json') as fin:
	for idx, line in enumerate(fin):
		if idx % 10000 == 0:
			print(idx)
		if idx >= tot * 0.8:
			continue
			
		js = json.loads(line)
		for W in js['text'].split():
			node2cnt[W] += 1

		for A0 in js['author']:
			A = 'AUTHOR_' + A0
			node2cnt[A] += 1

		if js['venue'] not in venue2id:
			venue2id[js['venue']] = str(len(venue2id))

with open(folder+'MeSH.json') as fin, open('mesh/network.dat', 'w') as fout:
	for idx, line in enumerate(fin):
		if idx % 10000 == 0:
			print(idx)
		if idx >= tot * 0.8:
			continue

		js = json.loads(line)

		P = 'PAPER_'+js['paper']
		left.add(P)
		
		# P-L		
		for L0 in js['MeSH']:
			L = 'LABEL_' + L0
			fout.write(P+' '+L+' 0 1 \n')
			right.add(L)

		# P-A
		for A0 in js['author']:
			A = 'AUTHOR_' + A0
			if node2cnt[A] >= thrs:
				fout.write(P+' '+A+' 1 1 \n')
				right.add(A)

		# P-V
		V = 'VENUE_' + venue2id[js['venue']]
		fout.write(P+' '+V+' 2 1 \n')
		right.add(V)

		# P-R
		for R0 in js['reference']:
			R = 'REFP_'+R0
			fout.write(P+' '+R+' 3 1 \n')
			right.add(R)

		# P-W
		words = js['text'].split()
		for W in words:
			if node2cnt[W] >= thrs:
				fout.write(P+' '+W+' 4 1 \n')
				right.add(W)

		# Wc-W
		for i in range(len(words)):
			Wi = words[i]
			if node2cnt[Wi] < thrs:
				continue
			for j in range(i-5, i+6):
				if j < 0 or j >= len(words) or j == i:
					break
				Wj = words[j]
				if node2cnt[Wj] < thrs:
					continue
				fout.write(Wj+' '+Wi+' 5 1 \n')
				left.add(Wj)
			
with open('mesh/left.dat', 'w') as fou1, open('mesh/right.dat', 'w') as fou2:
	for x in left:
		fou1.write(x+'\n')
	for x in right:
		fou2.write(x+'\n')