import faulthandler; faulthandler.enable()
import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logzero import logger

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model
from deepxml.match import MATCH

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('--reg', type=click.BOOL, default=False)

def main(data_cnf, model_cnf, mode, reg):
	yaml = YAML(typ='safe')
	data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
	model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
	model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
	emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
	logger.info(F'Model Name: {model_name}')

	if mode is None or mode == 'train':
		logger.info('Loading Training and Validation Set')
		train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
		if 'size' in data_cnf['valid']:
			random_state = data_cnf['valid'].get('random_state', 1240)
			train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
																			test_size=data_cnf['valid']['size'],
																			random_state=random_state)
		else:
			valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
		mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((train_labels, valid_labels)))
		train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
		labels_num = len(mlb.classes_)
		logger.info(F'Number of Labels: {labels_num}')
		logger.info(F'Size of Training Set: {len(train_x)}')
		logger.info(F'Size of Validation Set: {len(valid_x)}')

		edges = set()
		if reg:
			classes = mlb.classes_.tolist()
			with open(data_cnf['hierarchy']) as fin:
				for line in fin:
					data = line.strip().split()
					p = data[0]
					if p not in classes:
						continue
					p_id = classes.index(p)
					for c in data[1:]:
						if c not in classes:
							continue
						c_id = classes.index(c)
						edges.add((p_id, c_id))
			logger.info(F'Number of Edges: {len(edges)}')

		logger.info('Training')
		train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
								  model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
		valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=True),
								  model_cnf['valid']['batch_size'], num_workers=4)
		model = Model(network=MATCH, labels_num=labels_num, model_path=model_path, emb_init=emb_init, mode='train', reg=reg, hierarchy=edges,
					  **data_cnf['model'], **model_cnf['model'])
		model.train(train_loader, valid_loader, **model_cnf['train'])
		logger.info('Finish Training')

	if mode is None or mode == 'eval':
		logger.info('Loading Test Set')
		mlb = get_mlb(data_cnf['labels_binarizer'])
		labels_num = len(mlb.classes_)
		test_x, _ = get_data(data_cnf['test']['texts'], None)
		logger.info(F'Size of Test Set: {len(test_x)}')

		logger.info('Predicting')
		test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'], num_workers=4)
		if model is None:
			model = Model(network=MATCH, labels_num=labels_num, model_path=model_path, emb_init=emb_init, mode='eval',
						  **data_cnf['model'], **model_cnf['model'])
		scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
		logger.info('Finish Predicting')
		labels = mlb.classes_[labels]
		output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', scores, labels)

if __name__ == '__main__':
	main()
