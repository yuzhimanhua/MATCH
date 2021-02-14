# MATCH: Metadata-Aware Text Classification in A Large Hierarchy

This project focuses on metadata/hierarchy-aware extreme multi-label text classification.

## Installation

For training, GPUs are required. In our experiments, the code is run on two GeForce GTX 1080.

### Dependency
The code is written in Python 3.6. The dependencies are summarized in the file ```requirements.txt```. You can install them like this:

```
pip3 install -r requirements.txt
```
## Quick Start
To reproduce the results in our paper, you need to first download the [**datasets**](https://drive.google.com/file/d/1pn9WhPxIR4J7Wgm5_AJLgNHvTaMexDcC/view?usp=sharing). Two datasets are used in the paper: **MAG-CS** and **PubMed**. Once you unzip the downloaded file, you can see two folders, ```MAG/``` (corresponding to MAG-CS) and ```MeSH``` (corresponding to PubMed). You need to put these two folders under the repository main folder ```./```. Then you need to run the following scripts.

### Preprocessing
```
./preprocess.sh
```

### Training, Testing, and Evaluation
```
./run_models.sh
```
P@_k_ and NDCG@_k_ scores will be shown in the last several lines of the output. The prediction results (top-5 labels of each testing document) can be found in ```./predictions.txt```. For more detailed output (e.g., the trained model and the prediction scores), please refer to the **Running** section below.

## Data
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
Here, each paper is represented by its [Microsoft Academic Graph (MAG)](https://academic.microsoft.com/home) Paper ID (in both the "paper" field and the "reference" field); each author is represented by its MAG author ID; each label is represented by its MAG Field-of-Study ID. The "text" field is a sequence of words. Due to copyright issues, we represent each word as a number. Meanwhile, we provide a vocabulary file ```MAG/vocabulary.txt``` which maps each number back to its original word.
```
0	engineered
1	annexin
2	a5
3	variants
4	have
5	impaired
```
Using this vocabulary, you can recover the original text information for your own use.

**NOTE: If you would like to run our code on your own datasets, there is no need to represent each paper/author/word as a number. Just make sure (1) each paper/venue/author/word name does not have whitespace inside and (2) the "paper" field and the "reference" field are referring to the same namespace.**

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
Here, each paper (in the "paper" and "reference" fields) or author is still represented by its MAG ID. We also provide the [PubMed](https://pubmed.ncbi.nlm.nih.gov/) ID of each paper in the "PMID" field. Each label is represented by its [MeSH](https://meshb-prev.nlm.nih.gov/search) ID. We also provide a vocabulary file ```MeSH/vocabulary.txt``` for you to recover the original text information.

## Running
