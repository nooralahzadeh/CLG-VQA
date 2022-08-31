import h5py
import pickle
import argparse
from collections import defaultdict
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, entries, shuffle=False):
        self.corpus_path = corpus_path
        self.shuffle = shuffle

        self.img2entries = defaultdict(list)
        for e in entries:
            self.img2entries[str(e['image_id'])].append(e)

        self.num_imgs = len(self.img2entries)
        self.num_entries = len(entries)

    def __len__(self):
        return self.num_entries

    def __iter__(self):
        with h5py.File(self.corpus_path, 'r') as f:
            img_ids = set(f.keys())
            this_ids = list(img_ids.intersection(self.img2entries.keys()))
            assert len(this_ids) == self.num_imgs
            for i, img_id in enumerate(this_ids):
                if i == 0:
                    keys = list(f[img_id].keys())

                item = {}
                for k in keys:
                    item[k] = f[f'{img_id}/{k}'][()]
                item['img_id'] = img_id

                for e in self.img2entries[img_id]:
                    item['entry'] = e
                    yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    basedir = "/home/user/fnoora/iglue"
    srv = "/srv/scratch1/fnoora/gqa/vg_gqa_imgfeat"
    H5 = f"{srv}/vg_gqa_obj36-36.h5"
    LMDB = f"{srv}/volta/gqa-valid_boxes36_.lmdb"
    TEXT = f"{basedir}/datasets/gqa/annotations/val_target.pkl"
    parser.add_argument('--h5', type=str, default=H5)
    parser.add_argument('--lmdb', type=str, default=LMDB)
    parser.add_argument('--annotation', type=str, default=TEXT)
    args = parser.parse_args()

    source_fname = args.h5
    target_fname = args.lmdb
    entries = pickle.load(open(args.annotation, "rb"))

    ds = PretrainData(source_fname, entries)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, target_fname)
