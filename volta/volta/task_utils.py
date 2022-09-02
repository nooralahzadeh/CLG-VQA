# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).
# Copyright (c) 2022, Farhad Nooralahzadeh (@nooralahzadeh).


# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging,os
import _pickle as cPickle

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from transformers import AutoTokenizer
from volta.datasets._image_features_reader import ImageFeaturesH5Reader

class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
    def forward(self, output, targets):
        """
        Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class loss_kd_regularization(nn.Module):
    def __init__(self):
        super(loss_kd_regularization, self).__init__()

    def forward(self, outputs, labels, similarity) :
        """
        loss function for mannually-designed regularization: Tf-KD_{reg}
        """
        alpha = 0.1
        T = 20
        multiplier=1
        correct_prob = 0.99    # the probability for correct class in u(k)
        loss_CE = F.cross_entropy(outputs, torch.argmax(labels.long(), dim=1))
        K = outputs.size(1)

        teacher_soft = torch.ones_like(outputs).cuda()
        teacher_soft = teacher_soft * (1-correct_prob)/(K-1)  # p^d(k)
        for i in range(outputs.shape[0]):
            teacher_soft[i ,torch.argmax(labels,dim=1)[i]] = correct_prob
        loss_soft_regu = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1) * similarity)*multiplier

        KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

        return KD_loss


class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, teacher_rep, epoch):
        loss_CE = self.cross_entropy(outputs, torch.argmax(labels.long(), dim=1))

        if epoch>4:
            teacher_rep = teacher_rep.to(labels.device)
            cosine_loss = (1 - torch.cosine_similarity(F.softmax(outputs, dim=-1), F.softmax(teacher_rep, dim=-1),
                                                       dim=-1)).mean() # sum()
            return loss_CE + (cosine_loss*10)
        else:
            return loss_CE

class loss_kd_self(nn.Module):
    def __init__(self):
        super(loss_kd_self, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, teacher_outputs=None, epoch=0):
        """
        loss function for self training: Tf-KD_{self}
        "alpha": 0.1,
        "temperature": 20,
        """
        alpha = 0.1
        T = 20
        multiplier = 1
        loss_CE = self.cross_entropy(outputs, torch.argmax(labels.long(), dim=1))

        if epoch > 0:
            teacher_outputs = teacher_outputs.to(labels.device)
            #tb = torch.zeros(teacher_outputs.shape).to(teacher_outputs.device)
            #tb[(torch.arange(len(teacher_outputs)).unsqueeze(1), torch.topk(teacher_outputs, 10).indices)] = 1
            # teacher_outputs= teacher_outputs * (1- distance)
            p_top_k, idx_top_k = torch.topk(F.log_softmax(outputs/T, dim=-1), k=10)

            teacher_outputs = F.softmax(teacher_outputs[torch.arange(teacher_outputs.size(0)).unsqueeze(1), idx_top_k]/T,dim=-1)
            D_KL = self.KLD(p_top_k, teacher_outputs) * (T * T) * multiplier  # multiple is 1.0 in most of cases, some cases are 10 or 50
            # KD_loss =  (1. - alpha) * loss_CE + alpha * D_KL
            KD_loss = loss_CE.mean() + D_KL
            # if epoch==1:
            #     print("D_kL", D_KL, "KD_loss",KD_loss)
            return KD_loss
        else:
            return loss_CE


class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
        self.cross_entropy=nn.CrossEntropyLoss()
        self.mse=nn.MSELoss()
        self.multipler=10

    def forward(self, outputs, labels, teacher_outputs=None, epoch=0):
        loss_CE = self.cross_entropy(outputs, torch.argmax(labels.long(), dim=1))

        if epoch > 0:
            teacher_outputs = teacher_outputs.to(labels.device)

            p_top_k, idx_top_k = torch.topk(F.softmax(outputs, dim=-1), k=10)
            teacher_outputs = F.softmax(teacher_outputs[torch.arange(teacher_outputs.size(0)).unsqueeze(1), idx_top_k],dim=-1)
            semantic_loss = self.mse(p_top_k,teacher_outputs) * self.multipler # multiple is 1.0 in most of cases, some cases are 10 or 50

            student_teacher_loss = loss_CE + semantic_loss

            return student_teacher_loss
        else:
            return loss_CE

class cosine_loss(nn.Module):
    def __init__(self):
        super(cosine_loss, self).__init__()
        self.cross_entropy=nn.CrossEntropyLoss()
        self.cosine=nn.CosineSimilarity()
        self.multipler=10

    def forward(self, outputs, labels, teacher_outputs=None, epoch=0):
        loss_CE = self.cross_entropy(outputs, torch.argmax(labels.long(), dim=1))

        if epoch > 0:
            teacher_outputs = teacher_outputs.to(labels.device)

            p_top_k, idx_top_k = torch.topk(F.softmax(outputs, dim=-1), k=10)
            teacher_outputs = F.softmax(teacher_outputs[torch.arange(teacher_outputs.size(0)).unsqueeze(1), idx_top_k],dim=-1)
            semantic_loss = torch.sum(1- self.cosine(p_top_k,teacher_outputs),dim=-1) # multiple is 1.0 in most of cases, some cases are 10 or 50
            semantic_loss= self.multipler * semantic_loss.mean()
            student_teacher_loss = loss_CE + semantic_loss

            return student_teacher_loss
        else:
            return loss_CE
class LogitNormLoss(nn.Module):

    def __init__(self, t=0.01):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)

logger = logging.getLogger(__name__)


def triplet_loss(rank_scores, target, margin=0.2):
    scores = torch.sigmoid(rank_scores)
    pos = scores[:, :1]
    neg = scores[:, 1:]
    rank_loss = torch.clamp(margin + neg - pos, 0)
    return rank_loss.mean()

def LoadLoss(args, task_cfg, task_id):
    task = "TASK" + task_id
    loss_name = args.loss or task_cfg[task]["loss"]
    loss = LossMap[loss_name]
    return loss

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(), #LogitNormLoss(),
    "TripletLoss": triplet_loss,
    "Loss_Skd":loss_kd_self(),
    "LogitNormLoss": LogitNormLoss(),
    "Loss_Skd_Mse":mse_loss(),
    "Loss_Skd_Cosine":cosine_loss(),
}

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor, masks) -> torch.Tensor:
    b,l,_ = logits.size()
    targets = torch.arange(l).to(logits.device)
    targets = targets.repeat(b,1)
    targets = torch.where(masks==0, torch.tensor(-100).to(logits.device), targets) #  where it is 0 put -100 otherwise current
    return nn.functional.cross_entropy(logits, targets, ignore_index=-100)

def clip_loss(similarity: torch.Tensor, masks) -> torch.Tensor:
    image_loss = contrastive_loss(similarity, masks=masks)
    caption_loss = contrastive_loss(torch.transpose(similarity,2,1), masks=masks)
    return (caption_loss + image_loss) / 2.0



def ForwardModelsVal_t_1(device, task_id, batch, model):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, question_id, ixs , _ = batch
    with torch.no_grad():
        vil_prediction, _, _, _ = model(question, features, spatials, task_id,
                                                                           segment_ids, input_mask, image_mask)
    return vil_prediction


def ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id, ixs = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id, ixs, _ = batch

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        raise NotImplementedError("dialog process for validation")

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    with torch.no_grad():
        vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                           segment_ids, input_mask, image_mask)

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, torch.argmax(target.long(), dim=1))
        # loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return float(loss), float(batch_score), batch_size



def ForwardModelsTrain(config, task_cfg, device, task_id, batch, model, criterion):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id, ixs = batch
    else:
        # distance added
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id, ixs, distances = batch

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
        segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask)

    # for different task, we use different output to calculate the loss.

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":

        #semantic loss
        semantic_lambda = task_cfg[task_id]["semantic_lambda"]
        # # top-k : k=10 shows best
        p_top_k, idx_top_k = torch.topk(F.softmax(vil_prediction, dim=-1), k=10)
        semantic_loss = p_top_k * distances[torch.arange(distances.size(0)).unsqueeze(1), idx_top_k]
        semantic_loss = torch.mean(torch.sum(semantic_loss, dim=-1), dim=0)
        #semantic loss

        loss = criterion(vil_prediction, torch.argmax(target.long(), dim=1))
        loss = loss.mean() * target.size(1)
        loss += (semantic_lambda * semantic_loss.mean()) * target.size(1)


        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    return loss, batch_score



def LoadDataset(args, config, task_cfg, task_id, split="trainval"):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    batch_size = args.batch_size or task_cfg[task]["batch_size"]
    batch_size //= args.grad_acc_steps
    num_workers = args.num_workers
    num_val_workers = args.num_val_workers
    cache = args.cache or 5000
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())
        num_val_workers = int(num_val_workers / dist.get_world_size())
        cache = int(cache / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    from volta.datasets import DatasetMapTrain

    train_annotations_jsonpath = args.train_annotations_jsonpath or task_cfg[task]["train_annotations_jsonpath"]
    val_annotations_jsonpath = args.val_annotations_jsonpath or task_cfg[task]["val_annotations_jsonpath"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]


    data_format = task_cfg[task]["format"]
    dset_train, dl_train, task2num_iters = None, None, {}
    dset_val, dl_val = None, None
    if data_format == "serialized_lmdb":
        if "train" in split:
            dl_train = DatasetMapTrain[task_name+"Loader"](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=train_annotations_jsonpath,
                split=args.train_split or task_cfg[task]["train_split"],
                image_features_reader=feats_h5path1,
                gt_image_features_reader=None,
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                padding_index=tokenizer.pad_token_id,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                num_locs=config.num_locs,
                add_global_imgfeat=config.add_global_imgfeat,
                append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
                norm_embeddings=config.norm_embeddings,
                batch_size=batch_size,
                num_workers=num_workers,
                cache=cache,
                # New argumments
                seed = args.seed,
                semantic_dict_path = task_cfg[task]["semantic_dict_path"], # semantic loss parameters
                dict_path = task_cfg[task]["dictionary_path"], #code-switching parameters
                ratio = task_cfg[task]["ratio"], # code-switching parameters
                cross = task_cfg[task]["cross"], # code-switching parameters
                do_code_mixing = task_cfg[task]["code_mixing"], # code-switching parameters
                word_attributes_path = task_cfg[task]["word_attributes_path"],# code-switching parameters
                do_clip = task_cfg[task]["clip"], # clip
                objects_labels_path = task_cfg[task]["objects_labels_path"], # clip

            )
            task2num_iters = {task: len(dl_train)}
        if "val" in split:
            dl_val = DatasetMapTrain[task_name+"Loader"](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=val_annotations_jsonpath,
                split=args.val_split or task_cfg[task]["val_split"],
                image_features_reader=feats_h5path2,
                gt_image_features_reader=None,
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                padding_index=tokenizer.pad_token_id,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                num_locs=config.num_locs,
                add_global_imgfeat=config.add_global_imgfeat,
                append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
                norm_embeddings=config.norm_embeddings,
                batch_size=batch_size,
                num_workers=num_val_workers,
                cache=cache,
                # ME
                seed = args.seed,
                semantic_dict_path=task_cfg[task]["semantic_dict_path"],  # semantic loss parameters
                dict_path = task_cfg[task]["dictionary_path"],  # code-switching parameters
                ratio = task_cfg[task]["ratio"],  # code-switching parameters
                cross = task_cfg[task]["cross"],  # code-switching parameters
                word_attributes_path=task_cfg[task]["word_attributes_path"], # code-switching parameters
                do_code_mixing = False,  # code-switching parameters
                do_clip = False,  # clip
                objects_labels_path=task_cfg[task]["objects_labels_path"],  # clip
            )

    else:
        features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
        features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None
        if "train" in split:
            if args.train_features_lmdbpath:
                features_reader = ImageFeaturesH5Reader(args.train_features_lmdbpath, config, args.in_memory)
            else:
                features_reader = None
            dset_train = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=train_annotations_jsonpath,
                split=args.train_split or task_cfg[task]["train_split"],
                image_features_reader=features_reader or features_reader1,
                gt_image_features_reader=features_reader2,
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                padding_index=tokenizer.pad_token_id,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                num_locs=config.num_locs,
                add_global_imgfeat=config.add_global_imgfeat,
                append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            )
            if args.local_rank == -1:
                train_sampler = RandomSampler(dset_train)
            else:
                train_sampler = DistributedSampler(dset_train)
            dl_train = DataLoader(
                dset_train,
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=args.drop_last,
            )
            task2num_iters = {task: len(dl_train)}

        if "val" in split:
            eval_batch_size = args.eval_batch_size or task_cfg[task].get("eval_batch_size", batch_size)
            dset_val = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=val_annotations_jsonpath,
                split=args.val_split or task_cfg[task]["val_split"],
                image_features_reader=features_reader1,
                gt_image_features_reader=features_reader2,
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                padding_index=tokenizer.pad_token_id,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"],
                num_locs=config.num_locs,
                add_global_imgfeat=config.add_global_imgfeat,
                append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            )
            dl_val = DataLoader(
                dset_val,
                shuffle=False,
                batch_size=eval_batch_size,
                num_workers=num_val_workers,
                pin_memory=True,
                drop_last=args.drop_last,
            )

    return batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val


def LoadDatasetEval(args, config, task_cfg, task_id):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = args.val_features_lmdbpath or task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task].get("eval_batch_size", args.batch_size)
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    from volta.datasets import DatasetMapEval
    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg[task]["val_split"]

    val_annotations_jsonpath = args.val_annotations_jsonpath or task_cfg[task]["val_annotations_jsonpath"]

    if task_name.startswith("Retrieval"): # or task_name == "xFlickr":
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=val_annotations_jsonpath,
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=tokenizer.pad_token_id,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            num_subiters=args.num_subiters,
        )
    else:
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=val_annotations_jsonpath,
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            padding_index=tokenizer.pad_token_id,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )

    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_val_workers,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    task2num_iters = {task: len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores




def EvaluatingModel(config, task_cfg, device, task_id, batch, model, dataloader, criterion, results, others):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id, ixs = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id, ixs = batch

    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    with torch.no_grad():
        vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                           segment_ids, input_mask, image_mask)

    if task_cfg[task_id]["type"] == "VL-classifier":
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

        preds = torch.max(vil_prediction, 1)[1].data  # argmax
        labels = torch.max(target, 1)[1].data
        for i in range(preds.size(0)):
            results.append(
                {
                    "sentence": dataloader.dataset.entries[question_id[i].item()]["sentence"],
                    "prediction": preds[i].item(),
                    "label": labels[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return float(loss), float(batch_score), batch_size, results, others