import _pickle as cPickle
import spacy
import os
import numpy as np
path = os.path.join('/srv/scratch1/fnoora')
dataroot = "/home/user/fnoora/iglue/datasets/gqa/annotations"
label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
label2ans = cPickle.load(open(label2ans_path, "rb"))


nlp = spacy.load('en_core_web_lg')
from tqdm import tqdm

lbls = list(label2ans)
already_calculated = []
similarities = {}
for s in tqdm(lbls):
    source = nlp(s)
    for t in [l for l in lbls if l != s]:
        if (lbls.index(s), lbls.index(t)) in similarities or (lbls.index(t), lbls.index(s)) in similarities:
            continue
        else:
            target = nlp(t)
            sim = source.similarity(target)
            #print(s,t,1-sim)
            similarities[(lbls.index(s), lbls.index(t))] = 1-sim
            similarities[(lbls.index(t), lbls.index(s))] = 1-sim

with open(f'{path}/embedding_distance.pkl', 'wb') as handle:
    # pickle.dump(semantic_relations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cPickle.dump(similarities, handle)

