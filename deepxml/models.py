import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping

from deepxml.evaluation import get_p_1, get_p_3, get_p_5, get_n_1, get_n_3, get_n_5
from deepxml.optimizers import DenseSparseAdam

class Model(object):
	def __init__(self, network, model_path, mode, reg=False, hierarchy=None, gradient_clip_value=5.0, device_ids=None, **kwargs):
		self.model = nn.DataParallel(network(**kwargs).cuda(), device_ids=device_ids)
		self.loss_fn = nn.BCEWithLogitsLoss()
		self.model_path, self.state = model_path, {}
		os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
		self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
		self.optimizer = None

		self.reg = reg
		if mode == 'train' and reg:
			self.hierarchy = hierarchy
			self.lambda1 = 1e-8
			self.lambda2 = 1e-8 

	def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
		self.optimizer.zero_grad()
		self.model.train()
		scores = self.model(train_x)
		loss = self.loss_fn(scores, train_y)

		if self.reg:
			# Output Regularization
			probs = torch.sigmoid(scores)
			regs = torch.zeros(len(probs), len(self.hierarchy))
			for idx, tup in enumerate(self.hierarchy):
				p = tup[0]
				c = tup[1]
				regs[:,idx] = probs[:,c] - probs[:,p]
			loss += self.lambda1 * torch.sum(nn.functional.relu(regs)).item()

			# # Parameter Regularization
			# # Note: Adding this will make your model training slow
			# weights = self.model.module.plaincls.out_mesh_dstrbtn.weight
			# regs = torch.zeros(len(weights[0]), len(self.hierarchy)).cuda()
			# for idx, tup in enumerate(self.hierarchy):
			# 	p = tup[0]
			# 	c = tup[1]
			# 	regs[:,idx] = weights[p] - weights[c]
			# loss += self.lambda2 * 1/2 * torch.norm(regs, p=2) ** 2

		loss.backward()
		self.clip_gradient()
		self.optimizer.step(closure=None)
		return loss.item()	

	def predict_step(self, data_x: torch.Tensor, k: int):
		self.model.eval()
		with torch.no_grad():
			scores, labels = torch.topk(self.model(data_x), k)
			return torch.sigmoid(scores).cpu(), labels.cpu()

	def get_optimizer(self, **kwargs):
		self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)

	def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
			  nb_epoch=100, step=100, k=5, early=100, verbose=True, swa_warmup=None, **kwargs):
		self.get_optimizer(**({} if opt_params is None else opt_params))
		global_step, best_n5, e = 0, 0.0, 0
		print_loss = 0.0
		for epoch_idx in range(nb_epoch):
			if epoch_idx == swa_warmup:
				self.swa_init()
			for i, (train_x, train_y) in enumerate(train_loader, 1):
				global_step += 1
				loss = self.train_step(train_x, train_y.cuda())
				print_loss += loss
				if global_step % step == 0:
					self.swa_step()
					self.swap_swa_params()

					labels = []
					valid_loss = 0.0
					self.model.eval()
					with torch.no_grad():
						for (valid_x, valid_y) in valid_loader:
							logits = self.model(valid_x)
							valid_loss += self.loss_fn(logits, valid_y.cuda()).item()
							scores, tmp = torch.topk(logits, k)
							labels.append(tmp.cpu())
					valid_loss /= len(valid_loader)
					labels = np.concatenate(labels)

					targets = valid_loader.dataset.data_y
					p1, p3, p5, n3, n5 = get_p_1(labels, targets), get_p_3(labels, targets), get_p_5(labels, targets), get_n_3(labels, targets), get_n_5(labels, targets)
					if n5 > best_n5:
						self.save_model(True)
						best_n5, e = n5, 0
					else:
						e += 1
						if early is not None and e > early:
							return
					self.swap_swa_params()
					if verbose:
						log_msg = '%d %d train loss: %.7f valid loss: %.7f P@1: %.5f P@3: %.5f P@5: %.5f N@3: %.5f N@5: %.5f early stop: %d' % \
						(epoch_idx, i * train_loader.batch_size, print_loss / step, valid_loss, round(p1, 5), round(p3, 5), round(p5, 5), round(n3, 5), round(n5, 5), e)
						logger.info(log_msg)
						print_loss = 0.0

	def predict(self, data_loader: DataLoader, k=100, desc='Predict', **kwargs):
		self.load_model()
		scores_list, labels_list = zip(*(self.predict_step(data_x, k)
										 for data_x in tqdm(data_loader, desc=desc, leave=False)))
		return np.concatenate(scores_list), np.concatenate(labels_list)

	def save_model(self, last_epoch):
		if not last_epoch: return
		for trial in range(5):
			try:                
				torch.save(self.model.module.state_dict(), self.model_path)
				break
			except:
				print('saving failed')

	def load_model(self):
		self.model.module.load_state_dict(torch.load(self.model_path))

	def clip_gradient(self):
		if self.gradient_clip_value is not None:
			max_norm = max(self.gradient_norm_queue)
			total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
			self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
			if total_norm > max_norm * self.gradient_clip_value:
				logger.warn(F'Clipping gradients with total norm {round(total_norm, 5)} '
							F'and max norm {round(max_norm, 5)}')

	def swa_init(self):
		if 'swa' not in self.state:
			logger.info('SWA Initializing')
			swa_state = self.state['swa'] = {'models_num': 1}
			for n, p in self.model.named_parameters():
				swa_state[n] = p.data.cpu().detach()

	def swa_step(self):
		if 'swa' in self.state:
			swa_state = self.state['swa']
			swa_state['models_num'] += 1
			beta = 1.0 / swa_state['models_num']
			with torch.no_grad():
				for n, p in self.model.named_parameters():
					swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

	def swap_swa_params(self):
		if 'swa' in self.state:
			swa_state = self.state['swa']
			for n, p in self.model.named_parameters():
				p.data, swa_state[n] = swa_state[n].cuda(), p.data.cpu()

	def disable_swa(self):
		if 'swa' in self.state:
			del self.state['swa']
			