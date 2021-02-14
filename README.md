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
To reproduce the results in our paper, you need to first download the [**datasets**](https://drive.google.com/file/d/1ktIzp1LR2DN-SMwNm91nYdyEoqhDBAE3/view?usp=sharing). Two datasets are used in the paper: **MAG-CS** and **PubMed**. Once you unzip the downloaded file, you can see two folders, ```MAG/``` (corresponding to MAG-CS) and ```MeSH``` (corresponding to PubMed). You need to put these two folders under the repository main folder ```./```. Then you need to run the following scripts.

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
