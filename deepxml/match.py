import torch
import torch.nn as nn
import math
import json
import copy
import numpy as np


ACT2FN = {"relu": torch.nn.functional.relu}


class BertConfig(object):
	def __init__(self,
				 vocab_size_or_config_json_file,
				 hidden_size,
				 num_hidden_layers,
				 num_attention_heads,
				 intermediate_size,
				 hidden_act,
				 hidden_dropout_prob,
				 attention_probs_dropout_prob,
				 max_position_embeddings):
		if isinstance(vocab_size_or_config_json_file, str):
			with open(vocab_size_or_config_json_file, "r") as reader:
				json_config = json.loads(reader.read())
			for key, value in json_config.items():
				self.__dict__[key] = value
		elif isinstance(vocab_size_or_config_json_file, int):
			self.vocab_size = vocab_size_or_config_json_file
			self.hidden_size = hidden_size
			self.num_hidden_layers = num_hidden_layers
			self.num_attention_heads = num_attention_heads
			self.hidden_act = hidden_act
			self.intermediate_size = intermediate_size
			self.hidden_dropout_prob = hidden_dropout_prob
			self.attention_probs_dropout_prob = attention_probs_dropout_prob
			self.max_position_embeddings = max_position_embeddings
		else:
			raise ValueError("First argument must be either a vocabulary size (int)"
							 "or the path to a pretrained model config file (str)")

	@classmethod
	def from_dict(cls, json_object):
		config = BertConfig(vocab_size_or_config_json_file=-1)
		for key, value in json_object.items():
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		with open(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayerNorm(nn.Module):
	def __init__(self, config):
		super(BertLayerNorm, self).__init__()

	def forward(self, x):
		return x


class BertEmbeddings(nn.Module):
	def __init__(self, config, emb_init, emb_trainable):
		super(BertEmbeddings, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
											_weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
		self.word_embeddings.weight.requires_grad = emb_trainable
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

		self.LayerNorm = BertLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)

		embeddings = words_embeddings + position_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class BertSelfAttention(nn.Module):
	def __init__(self, config):
		super(BertSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_scores = attention_scores + attention_mask

		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer


class BertSelfOutput(nn.Module):
	def __init__(self, config):
		super(BertSelfOutput, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertAttention(nn.Module):
	def __init__(self, config):
		super(BertAttention, self).__init__()
		self.self = BertSelfAttention(config)
		self.output = BertSelfOutput(config)

	def forward(self, input_tensor, attention_mask):
		self_output = self.self(input_tensor, attention_mask)
		attention_output = self.output(self_output, input_tensor)
		return attention_output


class BertIntermediate(nn.Module):
	def __init__(self, config):
		super(BertIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.intermediate_act_fn = ACT2FN[config.hidden_act] \
			if isinstance(config.hidden_act, str) else config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BertOutput(nn.Module):
	def __init__(self, config):
		super(BertOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertLayer(nn.Module):
	def __init__(self, config):
		super(BertLayer, self).__init__()
		self.attention = BertAttention(config)
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(self, hidden_states, attention_mask):
		attention_output = self.attention(hidden_states, attention_mask)
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class BertEncoder(nn.Module):
	def __init__(self, config):
		super(BertEncoder, self).__init__()
		self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])    

	def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
		all_encoder_layers = []
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states, attention_mask)
			if output_all_encoded_layers:
				all_encoder_layers.append(hidden_states)
		if not output_all_encoded_layers:
			all_encoder_layers.append(hidden_states)
		return all_encoder_layers


class BertLastSelfAttention(nn.Module):
	def __init__(self, config, n_probes):
		super(BertLastSelfAttention, self).__init__()
		self.n_probes = n_probes
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask):
		mixed_query_layer = self.query(hidden_states[:, :self.n_probes, :])
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_scores = attention_scores + attention_mask

		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer
	
	
class LastLayer(nn.Module):
	def __init__(self, config, n_probes):
		super(LastLayer, self).__init__()        
		self.n_probes = n_probes
		self.selfattn = BertLastSelfAttention(config, n_probes)
		self.selfoutput = BertSelfOutput(config)        
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(self, input_tensor, attention_mask):
		self_output = self.selfattn(input_tensor, attention_mask)
		attention_output = self.selfoutput(self_output, input_tensor[:, :self.n_probes, :])        
		intermediate_output = self.intermediate(attention_output)
		context_vectors = self.output(intermediate_output, attention_output)
		batch_size = context_vectors.size(0)
		context_vectors = context_vectors.view(batch_size, -1)
		return context_vectors

	
class BaseBertModel(nn.Module):
	def __init__(self, hidden_size, n_layers, n_probes, n_aheads, intermediate_size, dropout, hidden_act="relu", src_max_len=500, 
				 padding_idx=0, vocab_size=None, emb_init=None, emb_trainable=True, bottleneck_dim=None, **kwargs):
		super(BaseBertModel, self).__init__()     
		self.initializer_range = 0.02
		self.padding_idx = padding_idx
		if emb_init is not None:
			if vocab_size is not None:
				assert vocab_size == emb_init.shape[0]
			if hidden_size is not None:
				assert hidden_size == emb_init.shape[1]
			vocab_size, hidden_size = emb_init.shape
		self.register_buffer('tok_cls', torch.LongTensor([vocab_size + i for i in range(n_probes)]))
		vocab_size = vocab_size + n_probes
		emb_init = np.vstack((emb_init, np.random.normal(loc=0.0, scale=self.initializer_range, size=(n_probes, hidden_size))))
		bertconfig = BertConfig(vocab_size, hidden_size, n_layers, n_aheads, intermediate_size, hidden_act=hidden_act, 
						hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout, 
						max_position_embeddings=src_max_len+n_probes)
		self.embeddings = BertEmbeddings(bertconfig, emb_init, emb_trainable)
		self.encoder = BertEncoder(bertconfig)
		self.lastlayer = LastLayer(bertconfig, n_probes)
		if bottleneck_dim is None:
			self.pooler = nn.Linear(hidden_size * n_probes, hidden_size * n_probes)
		else:
			self.pooler = nn.Linear(hidden_size * n_probes, bottleneck_dim)
		self.apply(self.init_bert_weights)
		
	def init_bert_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			nn.init.xavier_uniform_(module.weight)
		elif isinstance(module, BertLayerNorm):
			pass
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()
			
	def forward(self, raw_input_variables):
		cls_variables = self.tok_cls.expand(raw_input_variables.size(0), -1)
		input_variables = torch.cat((cls_variables, raw_input_variables), dim=1)
		attention_mask = input_variables != self.padding_idx

		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		embedding_output = self.embeddings(input_variables)
		encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=False)
		sequence_output = encoded_layers[-1]
		context_vectors = torch.nn.functional.elu(self.lastlayer(sequence_output, extended_attention_mask))
		context_vectors = torch.nn.functional.elu(self.pooler(context_vectors))
		
		return context_vectors
	
	
class PlainC(nn.Module):
	def __init__(self, labels_num, hidden_size, n_probes):
		super(PlainC, self).__init__()
		self.out_mesh_dstrbtn = nn.Linear(hidden_size * n_probes, labels_num)
		nn.init.xavier_uniform_(self.out_mesh_dstrbtn.weight)

	def forward(self, context_vectors):
		output_dstrbtn = self.out_mesh_dstrbtn(context_vectors)  
		return output_dstrbtn
	

class MATCH(nn.Module):
	def __init__(self, hidden_size, n_layers, n_aheads, intermediate_size, dropout, labels_num, n_probes, **kwargs):
		super(MATCH, self).__init__()
		self.tewp = BaseBertModel(hidden_size, n_layers, n_probes, n_aheads, intermediate_size,  dropout, **kwargs)
		self.plaincls = PlainC(labels_num, hidden_size, n_probes)
			
	def forward(self, input_variables):
		context_vectors = self.tewp(input_variables)
		logits = self.plaincls(context_vectors)
		return logits
