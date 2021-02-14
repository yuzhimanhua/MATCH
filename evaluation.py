import warnings
warnings.filterwarnings('ignore')

import click
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from deepxml.evaluation import get_p_1, get_p_3, get_p_5, get_n_1, get_n_3, get_n_5

@click.command()
@click.option('-r', '--results', type=click.Path(exists=True), help='Path of results.')
@click.option('-t', '--targets', type=click.Path(exists=True), help='Path of targets.')
@click.option('--train-labels', type=click.Path(exists=True), default=None, help='Path of labels for training set.')

def main(results, targets, train_labels):
	res, targets = np.load(results, allow_pickle=True), np.load(targets, allow_pickle=True)

	topk = 5
	with open('predictions.txt', 'w') as fout:
		for labels in res:
			fout.write(' '.join(labels[:topk])+'\n')

	mlb = MultiLabelBinarizer(sparse_output=True)
	targets = mlb.fit_transform(targets)
	print('Precision@1,3,5:', get_p_1(res, targets, mlb), get_p_3(res, targets, mlb), get_p_5(res, targets, mlb))
	print('nDCG@1,3,5:', get_n_1(res, targets, mlb), get_n_3(res, targets, mlb), get_n_5(res, targets, mlb))
	
if __name__ == '__main__':
	main()
