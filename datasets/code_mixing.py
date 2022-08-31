import random
import numpy as np
import glob,os

ratio=1
cross=0.9
do_code_mix=True


def load_worddict(dict_path):
    src2tgt = []
    tgts = []
    for file in glob.glob(f"{dict_path}/*.txt"):
        tgts.append(os.path.basename(os.path.normpath(file))[:2])
        src2tgt.append({})
        with open(file, encoding="utf8") as reader:
            raw = reader.readlines()
            for line in raw:
                try:
                    line = line[: -1] if line[-1] == "\n" else line
                    src, tgt = line.split("\t")
                except:
                    src, tgt = line.split(" ")

                if src not in src2tgt[-1]:
                    src2tgt[-1][src] = [tgt]
                else:
                    src2tgt[-1][src].append(tgt)
    return {"languages": tgts, "src2tgt": src2tgt}

wordDicts=load_worddict("/home/user/fnoora/iglue/datasets/dictionary/MUSE")


def do_code_mix(x, disable=False):
    # if not disable and self.code_mix and (self.cross >= np.random.rand()):
    if not disable and (cross >= np.random.rand()):
        lan = random.randint(0, len(wordDicts["languages"]) - 1)
        # lan = list(self.wordDicts["languages"]).index("bn")
        if x in wordDicts["src2tgt"][lan]:
            print(wordDicts["languages"][lan],x)
            print(wordDicts["src2tgt"][lan][x][random.randint(0, len(wordDicts["src2tgt"][lan][x]) - 1)])
            return wordDicts["src2tgt"][lan][x][random.randint(0, len(wordDicts["src2tgt"][lan][x]) - 1)]
        else:
            return x
    else:
        return x


def cross_list(question, selected_idx=None):
    mixed = []
    if selected_idx == None:
        selected_idx = [idx for idx in range(len(question.split()))]
    for idx, xx in enumerate(question.split()):
        if idx in selected_idx:
            mixed.append(do_code_mix(xx.lower(), not ratio >= np.random.rand()))
        else:
            mixed.append(xx)
    return " ".join(mixed)



question="Who is flying through the sky?"
question = cross_list(question)
question = question.replace(' ?', '?')
print(question)