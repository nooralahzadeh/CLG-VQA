
import numpy as np
import glob
import os
import json
import pickle
import nltk
from nltk.corpus import wordnet
import _pickle as cPickle
dataroot="/home/user/fnoora/iglue/datasets/gqa/annotations"
label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
label2ans = cPickle.load(open(label2ans_path, "rb"))



def get_syn_hyper(word):
    synonyms = []
    hypernyms = []
    hyponyms = []
    word_ = word.replace(" ", "_")
    synset = wordnet.synsets(word_)
    if len(synset) == 0:
        word_ = word.replace(" ", "-")

    for syn in wordnet.synsets(word_):
        for l in syn.lemma_names():
            synonyms.append(l)
        # print( list(set([w for s in syn.closure(lambda s:s.hypernyms()) for w in s.lemma_names()])))
        for hpe in syn.hypernyms():
            for h_n in hpe.lemma_names():
                hypernyms.append(h_n)
        for hpo in syn.hyponyms():
            for h_n in hpo.lemma_names():
                hyponyms.append(h_n)

    return list(set(synonyms)), list(set(hypernyms)), list(set(hyponyms))

train_labels_synset={}

for label in label2ans:
    syns,hyps,hpos=get_syn_hyper(label)
    train_labels_synset[label]={"syn":syns,"hyp":hyps,"hpo":hpos}

lbls=list(train_labels_synset.keys())
print(len(lbls))
semantic_relations={}
semantic_relations_index={}
for lbl in lbls:
    syn=[]
    hyp=[]
    hpo=[]
    for lbl2 in [l for l in lbls if l!=lbl]:
        if lbl2 in train_labels_synset[lbl]['syn']:
            syn.append(lbl2)
        elif lbl2 in train_labels_synset[lbl]['hyp']:
            hyp.append(lbl2)
        elif lbl2 in train_labels_synset[lbl]['hpo']:
            hpo.append(lbl2)
    semantic_relations[lbl]={"syn":syn,"hyp":hyp,"hpo":hpo}
    semantic_relations_index[label2ans.index(lbl)]={"syn":[label2ans.index(l) for l in syn],
                                                    "hyp":[label2ans.index(l) for l in hyp],
                                                    "hpo":[label2ans.index(l) for l in hpo]}

path = os.path.join('/srv/scratch1/fnoora')
with open(f'{path}/l2l_semantic_index.pkl', 'wb') as handle:
    cPickle.dump(semantic_relations_index,handle)