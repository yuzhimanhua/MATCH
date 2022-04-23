# MATCH: Metadata-Aware Text Classification in A Large Hierarchy

This repository contains the source code for [**MATCH: Metadata-Aware Text Classification in A Large Hierarchy**](https://arxiv.org/abs/2102.07349).

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Running](#running)
- [Running on New Datasets](#running-on-new-datasets)
- [Citation](#citation)

## Installation

For training, GPUs are required. In our experiments, the code is run on two GeForce GTX 1080.

### Dependency
The code is written in Python 3.6. The dependencies are summarized in the file ```requirements.txt```. You can install them like this:

```
pip3 install -r requirements.txt
```
## Quick Start
To reproduce the results in our paper, you need to first download the [**datasets**](https://drive.google.com/file/d/1pn9WhPxIR4J7Wgm5_AJLgNHvTaMexDcC/view?usp=sharing). Two datasets are used in the paper: **MAG-CS** and **PubMed**. Once you unzip the downloaded file, you can see two folders, ```MAG/``` (corresponding to MAG-CS) and ```MeSH/``` (corresponding to PubMed). You need to put these two folders under the repository main folder ```./```. Then you need to run the following scripts.

### Preprocessing
```
./preprocess.sh
```

### Training, Testing, and Evaluation
```
./run_models.sh
```
P@_k_ and NDCG@_k_ scores will be shown in the last several lines of the output. The prediction results (top-5 labels of each testing document) can be found in ```./predictions.txt```. For more detailed output (e.g., the trained model and the prediction scores), please refer to the [Running](#running) section below.

## Data
Two datasets are used in our paper.
|  | MAG-CS | PubMed | 
|--|--------|--------|
| \# Training Documents   | 564,340 | 718,837 |
| \# Validation Documents | 70,534  | 89,855  |
| \# Testing Documents    | 70,533  | 89,854  |
| \# Labels               | 15,809 (15,308 appear in training) | 17,963 (17,763 appear in training) |
| \# Labels / Doc         | 5.60    | 7.78    |
| Vocabulary Size         | **425,316\*** | 776,975 |
| \# Words / Doc          | 126.33  | 198.97  |
| \# Authors              | 818,927 | 2,201,919 |
| \# Venues               | 105     | 150     |
| \# Paper-Author Edges   | 2,274,546 | 5,989,142 |
| \# Paper-Venue Edges    | 705,407 | 898,546 |
| \# Paper-Paper Edges    | 1,518,466 | 4,455,702 |
| \# Edges in Taxonomy    | 27,288  | 22,842  |
| \# Layers of Taxonomy   | 6       | 15      |

**\*: There is a typo in original paper. Please refer to the number here.**

The datasets are provided in json format (```MAG/MAG.json``` and ```MeSH/MeSH.json```). Each line in the json file represent one document. 

### MAG-CS
The format of ```MAG/MAG.json``` is as follows:
```
{
  "paper": "2805510628",
  "venue": "UbiComp",
  "author": [
    "2684633850", "2807016802", "2717621310", "2807362790"
  ],
  "reference": [
    "2148837283"
  ],
  "text": "0 1 2 3 4 5 6 7 2 8 9 10 11 12 13 14 15 16 17 12 0 18 19 20 21 22 13 4 5 23 24 25 26 27 0 ...",
  "label": [
    "102602991", "311688", "74211669", "2775973920", "35578498", "2778505942", "120314980", "107457646", "44010500", "19012869"
  ]
}
```
Here, each paper is represented by its [Microsoft Academic Graph (MAG)](https://academic.microsoft.com/home) Paper ID (in both the "paper" field and the "reference" field); each author is represented by its MAG author ID; each label is represented by its MAG Field-of-Study ID. 

**NOTE: In both datasets, there are a few papers appearing in the "reference" field but not appearing as a record in the dataset. In other words, a paper Y may be in the "reference" field of a paper X, but there is no record in the dataset whose "paper" field is Y.**

The "text" field is a sequence of words. Due to copyright issues, we represent each word as a number. Meanwhile, we provide a vocabulary file ```MAG/vocabulary.txt``` which maps each number back to its original word.
```
0	engineered
1	annexin
2	a5
3	variants
4	have
5	impaired
```
Using this vocabulary, you can recover the original text information for your own use.

We also provide the mapping from each author/label ID to the corresponding name. The author mapping can be found in ```MAG/id2author.txt```.
```
7574581 Zdenek Krnoul
2404438944  Milos Zelezný
2490164490  Jan Novák
2656294450  Petr Císar
22492467  Adam Kilgarriff
```
The label mapping can be found in ```MAG/id2label.txt```.
```
10389098  batch_file
11045955  elgamal_encryption
13818915  2_3_tree
18781661  star_height
19044487  control_zone
```

**We divide ```MAG.json``` into ```train.json```, ```dev.json```, and ```test.json``` using an 80%-10%-10% split.** 

The labels in MAG are organized into a DAG-structured hierarachy. The hierarchy information is in ```MAG/taxonomy.txt```. Each line in a number of labels separated by whitespace. The first label is the parent label and the remaining ones are its children.
```
92111932  22965304
56317617  67422183
186429297 90240001
45384764  26336911
89720835  46359721  2780477985  73510573
```

**NOTE: If you would like to run our code on your own datasets, there is no need to represent each paper, word, or metadata instance (e.g., author, reference) as a number. Just make sure that (1) each paper, word, or metadata instance does not have whitespace inside and (2) the "paper" field and the "reference" field are referring to the same namespace.**

### PubMed
The format of ```MeSH/MeSH.json``` is as follows:
```
{
  "paper": "2951082630",
  "PMID": "28939614",
  "venue": "Journal_of_Cell_Biology",
  "author": [
    "2048690779", "2554001348"
  ],
  "reference": [
    "2009307035", "2194184864", "2166283261", "2031502025", "2111472436"
  ],
  "text": "43 230 11 231 25 6 232 233 104 234 58 235 48 236 237 43 233 11 238 239 20 21 234 58 ...",
  "label": [
    "D048429", "D000431", "D005838", "D013997", "D010641", "D012441", "D059585", "D008938", "D005947", "D004734"
  ]
}
```
Here, each paper (in the "paper" and "reference" fields) or author is still represented by its MAG ID. We also provide the [PubMed](https://pubmed.ncbi.nlm.nih.gov/) ID of each paper in the "PMID" field. Each label is represented by its [MeSH](https://meshb-prev.nlm.nih.gov/search) ID. 

Similarly, we divide ```MeSH.json``` into ```train.json```, ```dev.json```, and ```test.json``` using an 80%-10%-10% split. The vocabulary, author mapping, label mapping, and hierarchy information is in ```MeSH/vocabulary.txt```, ```MeSH/id2author.txt```, ```MeSH/id2label.txt```, and ```MeSH/taxonomy.txt```, respectively.

## Running

The [Quick Start](#quick-start) section should be enough to reproduce the results in out paper. Here are more details of running our code.

### Required Input Files
There are many input files in our provided ```MAG/``` and ```MeSH``` folders. Some of them are optional and describe more information about the datasets, while the following 5 input files are required to run the code:

**```train.json```, ```dev.json```, ```test.json```, ```taxonomy.txt```, and ```meta_dict.json```.**

The first 4 files have been introduced above. The last file, ```meta_dict.json```, describes the metadata fields you would like to use in ```train.json```, ```dev.json```, and ```test.json```. For example, for both MAG-CS and PubMed, ```meta_dict.json``` is
```
{"metadata": ["venue", "author", "reference"]}
```
Our code also support new datasets with **new metadata fields**. Please refer to [Running on New Datasets](#running-on-new-datasets) below.

### Embedding Pre-Training
In the dataset folders, we have provided the pre-trained embedding files ```MAG/MAG.joint.emb``` and ```MeSH/MeSH.joint.emb```. To rerun embedding pre-training:
```
cd joint/
unzip eigen-3.3.3.zip
unzip gsl.zip
make
./run.sh
```
The commands above will install 2 external packages, Eigen and GSL (first used in [PTE](https://github.com/mnqu/PTE)). After installing the package, if needed, users also need to modify the package path in the makefile.

Make sure the "dataset" in ```run.sh``` is correct before running (default is MAG). The output embedding will be in the corresponding dataset folder (e.g., ```MAG/MAG.joint.emb```). 

### Preprocessing
```
./preprocess.sh
```

### Training
The following script contains commands for both training and testing.
```
./run_models.sh
```

In ```run_models.sh```, you can see the following command for training:
```
PYTHONFAULTHANDLER=1 python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --mode train --reg 0
```
The dataset/model configuration files are ```configure/datasets/$DATASET.yaml``` and ```configure/models/$MODEL-$DATASET.yaml```. You can make changes in these files to tune some parameters (e.g., number of Transformer layers, number of \[CLS\] tokens, number of attention heads, etc.).

```--reg 0``` means the model does not use hypernymy regularization. If you need it, just change it to ```--reg 1```.

After training , the model will be saved in ```$DATASET/models/```.

### Testing
In ```run_models.sh```, you can see the following command for testing:
```
PYTHONFAULTHANDLER=1 python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --mode eval
```
After testing, the predicted labels (top-100) and the correpsonding probabilities will be in ```$DATASET/results/$MODEL-$DATASET-labels.npy``` and ```$DATASET/results/$MODEL-$DATASET-scores.npy```, respectively.

Then the following command is used for evaluation:
```
python evaluation.py \
--results $DATASET/results/$MODEL-$DATASET-labels.npy \
--targets $DATASET/test_labels.npy \
--train-labels $DATASET/train_labels.npy
```
As mentioned in [Quick Start](#quick-start), P@_k_ and NDCG@_k_ scores will be shown in the last several lines of the output. The prediction results (top-5 labels) can be found in ```./predictions.txt```.

## Running on New Datasets
To run our model on new datasets, you need to prepare the following things:

(1) Create a new dataset folder ```$DATASET/```.

(2) The training, validation, and testing files ```$DATASET/train.json```, ```$DATASET/dev.json```, and ```$DATASET/test.json```. Each line is a json record. **Each json record must have the fields "paper" (i.e., document id), "text", and "label". You can define your own metadata fields, but make sure each of them is either a list of strings (e.g., "author") or a single string value (e.g., "venue").**

(3) The metadata field file ```$DATASET/meta_dict.json```. You need to tell us your defined metadata fields. Please refer to ```MAG/meta_dict.json```.

(4) The hierarchy file ```$DATASET/taxonomy.txt```. Each line in a number of labels separated by whitespace. The first label is the parent label and the remaining ones are its children. Please refer to ```MAG/taxonomy.txt```.

(5) ```configure/datasets/$DATASET.yaml``` and ```configure/models/MATCH-$DATASET.yaml``` specifying the hyperparameters and file locations of your new dataset. Please refer to ```configure/datasets/MAG.yaml``` and ```configure/models/MATCH-MAG.yaml```.

## Citation
Our implementation is adapted from [CorNet](https://github.com/XunGuangxu/CorNet). If you find the implementation useful, please cite the following paper:
```
@inproceedings{zhang2021match,
  title={MATCH: Metadata-Aware Text Classification in A Large Hierarchy},
  author={Zhang, Yu and Shen, Zhihong and Dong, Yuxiao and Wang, Kuansan and Han, Jiawei},
  booktitle={WWW'21},
  pages={3246--3257},
  year={2021},
  organization={ACM / IW3C2}
}
```
