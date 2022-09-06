
This is the implementation of the approaches described in the paper:
> Farhad Nooralahzadeh and Rico Sennrich [Improving the Cross-Lingual Generalisation in Visual Question Answering](https://arxiv.org/abs/). arXiv 2022; abs/.

Our repository is based on [IGLUE](https://github.com/e-bug/iglue) and [VOLTA](https://github.com/e-bug/volta). We thank the authors for their wonderful open-source efforts.</br>
For sparse fine-tuning (SFT), we adopt the codebase of [BERT-Tickets](https://github.com/VITA-Group/BERT-Tickets) for our purpose.</br>
We provide the code for reproducing our results and fine-tuned models.


## Repository Setup

To set the environment, see "Repository Setup" in the [VOLTA's README](volta/README.md).


## Data
Following the [IGLUE](https://github.com/e-bug/iglue) repository:

[`datasets/`](datasets) contains the textual data for Visual Question Answering task (i.e. GQA amd xGQA).

Check out its [README](datasets/README.md) for links to preprocessed data  

Features extraction steps for each of dataset and backbone can be found under [`features_extraction/`](features_extraction). 


## Models

As a starting point, we use pretrained V&L models UC2 and M3P which can be downloaded from [ERDA](https://sid.erda.dk/sharelink/b1Rge0DwwW).

To reproduce our [results](results), you can download the fine-tuned models from [model repository](https://pub.cl.uzh.ch/users/fnoora/fine-tuned-checkpoint/).

- uc2.zip:</br>
-`uc2_zero_shot_sft_cdm_seed256` contains the checkpoint for `with_prior+sft+cdm` strategy using UC2 model</br>
- m3p.zip:</br>
-`m3p_zero_shot_sft_cdm_seed0` contains the checkpoint for `with_prior+sft+cdm` strategy using M3P model</br>
    
Model configuration files are stored in [`volta/config/`](volta/config). 


## Training

We provide the scripts we used to train and evaluate models in [`experiments/`](experiments):
- [`zero_shot/`](experiments/zero_shot): English fine-tuning and zero-shot/`translate test' evaluation

Task configuration files are stored in [config_tasks/](config_tasks).

### Training With prior:

Set `code_mixing: False` in [config_tasks/](config_tasks) for
    `iglue_trainval_tasks_boxes.dtu.yml` and 
    `iglue_trainval_tasks_X101.dtu.yml`

##### WordNet
We extract the WordNet relations among the labels using [nltk](https://www.nltk.org/). You can find the code at [extract wordnet relation](volta/extract_wn_rel.py).

Download the `l2l_semantic_index.pkl` file from [semantic dict repository](https://pub.cl.uzh.ch/users/fnoora/semantic_dict/) 

Set `semantic_dict_path: ../l2l_semantic_index.pkl` in [config_tasks/](config_tasks) for
    `iglue_trainval_tasks_boxes.dtu.yml` and
    `iglue_trainval_tasks_X101.dtu.yml`


##### Word Embeddings
We extract the word embeddings distance among the labels using [spaCy](https://spacy.io/). You can find the code at [extract embedding distance](volta/extract_emb_dist.py).</br>

You can Download the `embedding_distance.pkl` file from [semantic dict repository](https://pub.cl.uzh.ch/users/fnoora/semantic_dict/) 

Set `semantic_dict_path: ../embedding_distance.pkl` in [config_tasks/](config_tasks) for
    `iglue_trainval_tasks_boxes.dtu.yml` and
    `iglue_trainval_tasks_X101.dtu.yml`

```bash
source train.dtu.sh 0 <path_to_directory_of_pretrained_vl_model> <path_to_directory_for_fine_tuned_model>
```
### Training With prior + CDM:
Set `code_mixing: True` in [config_tasks/](config_tasks) for
    `iglue_trainval_tasks_boxes.dtu.yml` and
    `iglue_trainval_tasks_X101.dtu.yml`

### Training With prior + SFT:
Set `code_mixing: False` in [config_tasks/](config_tasks) for 
        `iglue_trainval_tasks_boxes.dtu.yml` and 
        `iglue_trainval_tasks_X101.dtu.yml`
 - Step_0: pruning
```bash
source train.dtu.pruned.sh 0 <path_to_directory_of_pretrained_vl_model> <path_to_directory_for_pruned_model>
```
 - Step_1: fine-tuning 
```bash
source train.dtu.sft.sh 0 <path_to_directory_of_pretrained_vl_model> <path_to_directory_of_pruned_model> <path_to_directory_for_fine_tuned_model> 
```
### Training With prior + SFT + CDM:
Set `code_mixing: True` in [config_tasks/](config_tasks) --> `iglue_trainval_tasks_boxes.dtu.yml` and  `iglue_trainval_tasks_X101.dtu.yml`
* Step_0: pruning
```bash
source train.dtu.pruned.sh 0 <path_to_directory_of_pretrained_vl_model> <path_to_directory_for_pruned_model>
```
* Step_1: fine-tuning
```bash
source train.dtu.sft.sh 0 <path_to_directory_of_pretrained_vl_model> <path_to_directory_of_pruned_model> <path_to_directory_for_fine_tuned_model>
 ```

## Evaluation

```bash 
source test.dtu.sh  <path_to_directory_of_fine_tuned_model> <name_of_fine-tuned-model>
```

The result of our best models can be found in [results](results).

### Analysis
The results in the `Further Analysis` section can be reproduced by [analysis.ipynb](volta/analysis.ipynb).
## License

This work is licensed under the MIT license.
Third-party software and data are subject to their respective licenses. <br>

## Citation
If you find this repository useful in your work, you can cite the following paper:

```
@article{nooralahzadeh-sennrich-2022,
    title = "Improving the Cross-Lingual Generalisation in Visual Question Answering",
    author = "Nooralahzadeh, Farhad  and
      Sennrich, Rico",,
    journal = "arXiv preprint ...."
    year = "2022",
    url = "https://arxiv.org/abs/...",
}
```
