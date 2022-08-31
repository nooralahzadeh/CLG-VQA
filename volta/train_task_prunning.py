# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune

# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule

from volta.config import BertConfig, M3PConfig
from volta.optimization import RAdam
from volta.encoders import BertForVLTasks, M3PForVLTasks
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume, save_prunned
from volta.task_utils import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def pruning_model_uc2(model, px, embeddings=False, global_pruning=False, cls=False, bias=False):
    parameters_to_prune = []

    if embeddings:
        parameters_to_prune.append('bert.embeddings.word_embeddings')
        parameters_to_prune.append('bert.embeddings.image_embeddings')

    for ii in range(24): # 24
        if ii % 2 == 0:
            parameters_to_prune.append('bert.encoder.layer.%d.attention_self.query' % (ii))
            parameters_to_prune.append('bert.encoder.layer.%d.attention_self.key' % (ii))
            parameters_to_prune.append('bert.encoder.layer.%d.attention_self.value' % (ii))
            parameters_to_prune.append('bert.encoder.layer.%d.attention_output.dense' % (ii))
            #parameters_to_prune.append((model.bert.encoder.layer[ii].attention_output.LayerNorm, 'weight'))
        if ii > 0 and ii % 2 ==1:
            parameters_to_prune.append('bert.encoder.layer.%d.intermediate.dense' %(ii))
            parameters_to_prune.append('bert.encoder.layer.%d.output.dense' %(ii))
            #parameters_to_prune.append((model.bert.encoder.layer[ii].output.LayerNorm,'weight'))
    parameters_to_prune.append('bert.t_pooler.dense')

    if cls:
        for ii in [0,2,3]:
           parameters_to_prune.append('clfs_dict.TASK15.logit_fc.%d' % (ii))

    if global_pruning:
        parameters_to_prune_global = []
        for name, module in model.named_modules():
            if name in parameters_to_prune:
                parameters_to_prune_global.append((module,'weight'))
                if bias:
                    parameters_to_prune_global.append((module,'bias'))


        parameters_to_prune_global = tuple(parameters_to_prune_global)

        prune.global_unstructured(
            parameters_to_prune_global,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
    else:

        for name, module in model.named_modules():
            if name in parameters_to_prune:
                prune.l1_unstructured(module, 'weight', amount=px)
                if bias:
                    prune.l1_unstructured(module, 'bias', amount=px)
def see_weight_rate_uc2(model, embedding=False, cls=False, bias=False):
    sum_list = 0
    zero_sum = 0
    if embedding:
        sum_list = sum_list + float(model.state_dict()[f'bert.embeddings.word_embeddings.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.embeddings.word_embeddings.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.embeddings.image_embeddings.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.embeddings.image_embeddings.weight_mask'] == 0))

    for ii in range(24):
        if ii % 2 == 0:
            #model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'
            sum_list = sum_list + float(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.query.weight_mask'].nelement())
            zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.query.weight_mask'] == 0))

            sum_list = sum_list + float(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.key.weight_mask'].nelement())
            zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.key.weight_mask'] == 0))

            sum_list = sum_list + float(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.value.weight_mask'].nelement())
            zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.value.weight_mask'] == 0))

            sum_list = sum_list + float(model.state_dict()[f'bert.encoder.layer.{ii}.attention_output.dense.weight_mask'].nelement())
            zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_output.dense.weight_mask'] == 0))


        if ii>0  and ii % 2 ==1:
            sum_list = sum_list + float(model.state_dict()[f'bert.encoder.layer.{ii}.intermediate.dense.weight_mask'].nelement())
            zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.intermediate.dense.weight_mask'] == 0))
            sum_list = sum_list + float(model.state_dict()[f'bert.encoder.layer.{ii}.output.dense.weight_mask'].nelement())
            zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.output.dense.weight_mask'] == 0))

    if cls:
       for ii in [0,2,3]:
           sum_list = sum_list + float(model.state_dict()[f'clfs_dict.TASK15.logit_fc.{ii}.weight_mask'].nelement())
           zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'clfs_dict.TASK15.logit_fc.{ii}.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.t_pooler.dense.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.t_pooler.dense.weight_mask'] == 0))

    if bias:
        for ii in range(24):
            if ii % 2 == 0:
                # model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'
                sum_list = sum_list + float(
                    model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.query.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.query.bias_mask'] == 0))

                sum_list = sum_list + float(
                    model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.key.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.key.bias_mask'] == 0))

                sum_list = sum_list + float(
                    model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.value.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_self.value.bias_mask'] == 0))

                sum_list = sum_list + float(
                    model.state_dict()[f'bert.encoder.layer.{ii}.attention_output.dense.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.attention_output.dense.bias_mask'] == 0))



            if ii > 0 and ii % 2 == 1:
                sum_list = sum_list + float(
                    model.state_dict()[f'bert.encoder.layer.{ii}.intermediate.dense.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.intermediate.dense.bias_mask'] == 0))
                sum_list = sum_list + float(
                    model.state_dict()[f'bert.encoder.layer.{ii}.output.dense.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'bert.encoder.layer.{ii}.output.dense.bias_mask'] == 0))

        if cls:
            for ii in [0, 2, 3]:
                sum_list = sum_list + float(
                    model.state_dict()[f'clfs_dict.TASK15.logit_fc.{ii}.bias_mask'].nelement())
                zero_sum = zero_sum + float(
                    torch.sum(model.state_dict()[f'clfs_dict.TASK15.logit_fc.{ii}.bias_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.t_pooler.dense.bias_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.t_pooler.dense.bias_mask'] == 0))
    return 100 * zero_sum / sum_list

def rewind_uc2(pre_weight, model, embeddings=False, cls=False, bias=False):
    recover_dict = {}
    name_list = []

    if embeddings:
        name_list.append(f'{model}bert.embeddings.word_embeddings.weight')
        name_list.append(f'{model}bert.embeddings.image_embeddings.weight')

    for ii in range(24):
        if ii % 2 == 0:
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.query.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.key.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.value.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.dense.weight')
            # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.LayerNorm.weight')
            #
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.v_query.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.v_key.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.v_value.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.v_dense.weight')
            # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.v_LayerNorm.weight')



        if ii>0 and  ii % 2 ==1:
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.intermediate.dense.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.dense.weight')
            # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.LayerNorm.weight')
            # #
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.intermediate.v_dense.weight')
            name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.v_dense.weight')
            # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.v_LayerNorm.weight')
    if cls:
        for ii in [0,2,3]:
            #clfs_dict.TASK15.logit_fc.{ii}'
            name_list.append(f'{model}clfs_dict.TASK15.logit_fc.' + str(ii) + '.weight')

    name_list.append(f'{model}bert.t_pooler.dense.weight')

    if bias:
        for ii in range(24):
            if ii % 2 == 0:
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.query.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.key.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.value.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.dense.bias')
                # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.LayerNorm.weight')
                #
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.v_query.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.v_key.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_self.v_value.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.v_dense.bias')
                # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.attention_output.v_LayerNorm.weight')

            if ii > 0 and ii % 2 == 1:
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.intermediate.dense.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.dense.bias')
                # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.LayerNorm.weight')
                #
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.intermediate.v_dense.bias')
                name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.v_dense.bias')
                # # name_list.append(f'{model}bert.encoder.layer.' + str(ii) + '.output.v_LayerNorm.weight')
        if cls:
            for ii in [0, 2, 3]:
                # clfs_dict.TASK15.logit_fc.{ii}'
                name_list.append(f'{model}clfs_dict.TASK15.logit_fc.' + str(ii) + '.bias')
        name_list.append(f'{model}bert.t_pooler.dense.bias')

    for key in pre_weight.keys():
        if 'bert' in key or 'clfs_dict' in key:
            if key in name_list:
                new_key = key + '_orig'
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict

def pruning_model_m3p(model, px, global_pruning=False):
    parameters_to_prune = []

    for ii in range(12): #13
        parameters_to_prune.append('bert.encoder.attentions.%d.q_lin' % (ii))
        parameters_to_prune.append('bert.encoder.attentions.%d.k_lin' % (ii))
        parameters_to_prune.append('bert.encoder.attentions.%d.v_lin' % (ii))
        parameters_to_prune.append('bert.encoder.attentions.%d.out_lin' % (ii))

        parameters_to_prune.append('bert.encoder.ffns.%d.lin1' % (ii))
        parameters_to_prune.append('bert.encoder.ffns.%d.lin2' % (ii))

        parameters_to_prune.append('bert.encoder.encoder_attn.%d.q_lin' % (ii))
        parameters_to_prune.append('bert.encoder.encoder_attn.%d.k_lin' % (ii))
        parameters_to_prune.append('bert.encoder.encoder_attn.%d.v_lin' % (ii))
        parameters_to_prune.append('bert.encoder.encoder_attn.%d.out_lin' % (ii))

    for ii in range(2):
        parameters_to_prune.append('bert.encoder.latent_transforms.%d.x_to_mu' % (ii))
        parameters_to_prune.append('bert.encoder.latent_transforms.%d.x_to_logvar' % (ii))
        parameters_to_prune.append('bert.encoder.latent_transforms.%d.out_dense' % (ii))
        parameters_to_prune.append('bert.encoder.original_transforms.%d.dense' % (ii))
        parameters_to_prune.append('bert.encoder.original_transforms.%d.dense_mu' % (ii))


    parameters_to_prune.append('bert.encoder.pooled_layer.dense')
    parameters_to_prune.append('bert.encoder.seq_relationship')
    parameters_to_prune.append('bert.encoder.pooled_layer2.dense')
    parameters_to_prune.append('bert.encoder.seq_relationship2')
    parameters_to_prune.append('bert.encoder.mrfr_dense')
    parameters_to_prune.append('bert.encoder.transformer_obj.dense')

    if global_pruning:
        parameters_to_prune_global = []
        for name, module in model.named_modules():
            if name in parameters_to_prune:
                parameters_to_prune_global.append((module,'weight'))


        parameters_to_prune_global = tuple(parameters_to_prune_global)

        prune.global_unstructured(
            parameters_to_prune_global,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
    else:
        for name, module in model.named_modules():
            if name in parameters_to_prune:
                prune.l1_unstructured(module, 'weight', amount=px)

def see_weight_rate_m3p(model):
    sum_list = 0
    zero_sum = 0

    for ii in range(12):
        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.attentions.{ii}.q_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.attentions.{ii}.q_lin.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.attentions.{ii}.k_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.attentions.{ii}.k_lin.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.attentions.{ii}.v_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.attentions.{ii}.v_lin.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.attentions.{ii}.out_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.attentions.{ii}.out_lin.weight_mask'] == 0))

        #####
        sum_list = sum_list + float(
            model.state_dict()[f'bert.encoder.ffns.' + str(ii) + '.lin1.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.ffns.' + str(ii) + '.lin1.weight_mask'] == 0))

        sum_list = sum_list + float(
            model.state_dict()[f'bert.encoder.ffns.' + str(ii) + '.lin2.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.ffns.' + str(ii) + '.lin2.weight_mask'] == 0))

        ###
        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.q_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.q_lin.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.k_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.k_lin.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.v_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.v_lin.weight_mask'] == 0))

        sum_list = sum_list + float(
            model.state_dict()[f'bert.encoder.encoder_attn.{ii}.out_lin.weight_mask'].nelement())
        zero_sum = zero_sum + float(
            torch.sum(model.state_dict()[f'bert.encoder.encoder_attn.{ii}.out_lin.weight_mask'] == 0))

    for ii in range(2):
        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.latent_transforms.' + str(ii) + '.x_to_mu.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.latent_transforms.' + str(ii) + '.x_to_mu.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.latent_transforms.' + str(ii) + '.x_to_logvar.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.latent_transforms.' + str(ii) + '.x_to_logvar.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.latent_transforms.' + str(ii) + '.out_dense.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.latent_transforms.' + str(ii) + '.out_dense.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.original_transforms.' + str(ii) + '.dense.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.original_transforms.' + str(ii) + '.dense.weight_mask'] == 0))

        sum_list = sum_list + float(model.state_dict()[f'bert.encoder.original_transforms.' + str(ii) + '.dense_mu.weight_mask'].nelement())
        zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.original_transforms.' + str(ii) + '.dense_mu.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.encoder.pooled_layer.dense.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.pooled_layer.dense.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.encoder.seq_relationship.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.seq_relationship.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.encoder.pooled_layer2.dense.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.pooled_layer2.dense.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.encoder.seq_relationship2.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.seq_relationship2.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.encoder.mrfr_dense.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.mrfr_dense.weight_mask'] == 0))

    sum_list = sum_list + float(model.state_dict()[f'bert.encoder.transformer_obj.dense.weight_mask'].nelement())
    zero_sum = zero_sum + float(torch.sum(model.state_dict()[f'bert.encoder.transformer_obj.dense.weight_mask'] == 0))

    return 100 * zero_sum / sum_list
def rewind_m3p(pre_weight, model):
    recover_dict = {}
    name_list = []


    for ii in range(12):
        name_list.append(f'{model}bert.encoder.attentions.' + str(ii) + '.q_lin.weight')
        name_list.append(f'{model}bert.encoder.attentions.' + str(ii) + '.k_lin.weight')
        name_list.append(f'{model}bert.encoder.attentions.' + str(ii) + '.v_lin.weight')
        name_list.append(f'{model}bert.encoder.attentions.' + str(ii) + '.out_lin.weight')
        ###
        name_list.append(f'{model}bert.encoder.ffns.' + str(ii) + '.lin1.weight')
        name_list.append(f'{model}bert.encoder.ffns.' + str(ii) + '.lin2.weight')
        ###
        name_list.append(f'{model}bert.encoder.encoder_attn.' + str(ii) + '.q_lin.weight')
        name_list.append(f'{model}bert.encoder.encoder_attn.' + str(ii) + '.k_lin.weight')
        name_list.append(f'{model}bert.encoder.encoder_attn.' + str(ii) + '.v_lin.weight')
        name_list.append(f'{model}bert.encoder.encoder_attn.' + str(ii) + '.out_lin.weight')

    for ii in range(2):
        ###
        name_list.append(f'{model}bert.encoder.latent_transforms.' + str(ii) + '.x_to_mu.weight')
        name_list.append(f'{model}bert.encoder.latent_transforms.' + str(ii) + '.x_to_logvar.weight')
        name_list.append(f'{model}bert.encoder.latent_transforms.' + str(ii) + '.out_dense.weight')
        name_list.append(f'{model}bert.encoder.original_transforms.' + str(ii) + '.dense.weight')
        name_list.append(f'{model}bert.encoder.original_transforms.' + str(ii) + '.dense_mu.weight')

    name_list.append(f'{model}bert.encoder.pooled_layer.dense.weight')
    name_list.append(f'{model}bert.encoder.seq_relationship.weight')
    name_list.append(f'{model}bert.encoder.pooled_layer2.dense.weight')
    name_list.append(f'{model}bert.encoder.seq_relationship2.weight')
    name_list.append(f'{model}bert.encoder.mrfr_dense.weight')
    name_list.append(f'{model}bert.encoder.transformer_obj.dense.weight')


    for key in pre_weight.keys():
        if 'bert.encoder' in key :
            if key in name_list:
                new_key = key + '_orig'
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict


def custom_replace(tensor, on_zero, on_non_zero):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--is_m3p", action='store_true', default=False,
                        help="Use M3P.")
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    parser.add_argument("--save_best_only", default=False, action="store_true")
    parser.add_argument("--save_every_num_epochs", default=1, type=int)
    # Task
    parser.add_argument("--train_split", default="", type=str)
    parser.add_argument("--val_split", default="", type=str)
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--train_annotations_jsonpath", default="", type=str,
                        help="train_annotations_jsonpath")
    parser.add_argument("--val_annotations_jsonpath", default="", type=str,
                        help="val_annotations_jsonpath")
    parser.add_argument("--train_features_lmdbpath", default="", type=str)
    # parser.add_argument("--uc2_retrieval", default=False, action="store_true")
    # Text
    parser.add_argument("--do_lower_case", action='store_true', default=False,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Training
    parser.add_argument("--num_epoch", default=None, type=int,
                        help="Max number of training epochs to perform.")
    parser.add_argument("--optim_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    parser.add_argument("--batch_size", default=None, type=int,
                        help="overwrites the config_tasks batch size")
    parser.add_argument("--eval_batch_size", default=None, type=int,
                        help="overwrites the config_tasks batch size")
    parser.add_argument("--max_val_batches", default=-1, type=int)
    parser.add_argument("--loss", default="", type=str,
                        help="alternative loss name")
    parser.add_argument("--eval_steps", default=sys.maxsize, type=int,
                        help="when to evaluate model")
    parser.add_argument("--cache", default=5000, type=int)
    # Scheduler
    parser.add_argument("--lr", default=None, type=float,
                        help="overwrites the config_tasks learning rate")
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--num_val_workers", type=int, default=2)
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    if args.is_m3p:
        config = M3PConfig.from_json_file(args.config_file)
    else:
        config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    base_lr = args.lr or task_cfg[task]["lr"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timestamp = (task_name + "_" + args.config_file.split("/")[1].split(".")[0] + prefix)
    save_path = os.path.join(args.output_dir, timestamp)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val = LoadDataset(args, config, task_cfg, args.task)

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    tb_logger = tbLogger(logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Model
    if "roberta" in args.bert_model:
        config.model = "roberta"
    if config.image_embeddings == "m3p":
        model = M3PForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
    else:
        model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])


    if task_cfg[task].get("embed_clf", None):
        logger.info('Initializing classifier weight for %s from pretrained word embeddings...' % task)
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if 'bert.embeddings.word_embeddings.weight' in k:
                word_embeddings = v.detach().clone()
                break
        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith('clfs_dict.%s.logit_fc.3' % task):
                module.weight.data = answers_word_embed_tensor.to(device=module.weight.data.device)
    # # FIXME
    # if args.uc2_retrieval and task in {"TASK21",}:
    #     if default_gpu:
    #         print("Initializing classifier from pretrained model...")
    #     model.init_output(args.from_pretrained, task)  # pretrain ITM head is different from ranking head

    # Optimization details
    freeze_layers(model)
    criterion = LoadLoss(args, task_cfg, args.task)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=base_lr,
                          eps=args.adam_epsilon,
                          betas=args.adam_betas,
                          correct_bias=args.adam_correct_bias)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    num_train_optim_steps = (task2num_iters[task] * args.optim_train_epochs // args.grad_acc_steps)
    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
    if args.lr_scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, max_score = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Move to GPU(s)
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Train

    # Save starting model
    #save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu, -1)

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        print("***** Running training *****")
        print("  Num Iters: ", task2num_iters[task])
        print("  Batch size: ", batch_size)
        print("  Num steps: %d" % num_train_optim_steps)

    # Train
    # store the original weights
    bias= False
    embeddings_to_prune = False
    cls = False
    global_pruning = True

    orig = rewind_m3p(model.state_dict(), 'module.' if n_gpu > 1 else '') if args.is_m3p \
        else rewind_uc2(model.state_dict(), 'module.' if n_gpu > 1 else '', embeddings=embeddings_to_prune, cls=cls, bias=bias)

    pruning_step = 0

    #print("starting pruning before epoch")
    #
    # pruning_model(model.module if hasattr(model, "module") else model, 0.1, embeddings=embeddings_to_prune,cls=cls)
    # rate_weight_equal_zero = see_weight_rate(model.module if hasattr(model, "module") else model,
    #                                          embedding=embeddings_to_prune,cls=cls)
    # pruning_step += 1
    # print('zero_rate = ', rate_weight_equal_zero)
    #
    # print('rewinding')
    # model_dict = model.state_dict()
    # model_dict.update(orig)
    # model.load_state_dict(model_dict)

    scores = 0
    max_epoch = args.num_epoch or task_cfg[task]['num_epoch']
    for epoch_id in tqdm(range(start_epoch, max_epoch), desc="Epoch"):
        model.train()
        # from pudb import set_trace; set_trace()
        for step, batch in enumerate(dl_train):
            iter_id = start_iter_id + step // args.grad_acc_steps + (epoch_id * len(dl_train))
            loss, score = ForwardModelsTrain(config, task_cfg, device, task, batch, model, criterion)
            scores += score

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:

                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()

                if global_step < warmup_steps or args.lr_scheduler == "warmup_linear":
                    scheduler.step()

                model.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train(epoch_id, iter_id, float(loss), float(scores / args.grad_acc_steps),
                                         optimizer.param_groups[0]["lr"], task, "train")
                scores = 0

            if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrain()

            # Decide whether to evaluate task
            if iter_id != 0 and iter_id % (args.eval_steps - 1) == 0:
                score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu,
                                 tb_logger)
                if score > max_score:
                    max_score = score
                    save_prunned(save_path, logger, iter_id, model, optimizer, scheduler,
                         global_step, tb_logger, default_gpu, max_score, is_best=True)

        print("starting pruning before at epoch %d" %(epoch_id))

        if args.is_m3p:
            pruning_model_m3p(model.module if hasattr(model, "module") else model, 0.1, global_pruning=global_pruning)
            rate_weight_equal_zero = see_weight_rate_m3p(model.module if hasattr(model, "module") else model)
        else:
            pruning_model_uc2(model.module if hasattr(model, "module") else model, 0.1, global_pruning=global_pruning, embeddings=embeddings_to_prune, cls=cls, bias=bias)
            rate_weight_equal_zero = see_weight_rate_uc2(model.module if hasattr(model, "module") else model,
                                                         embedding=embeddings_to_prune, cls=cls, bias=bias)
        pruning_step += 1
        print('zero_rate = ', rate_weight_equal_zero)

        print('rewinding')
        model_dict = model.state_dict()
        model_dict.update(orig)
        model.load_state_dict(model_dict)
        ########### zero masking ############################################################

        # mask_dict = {}
        # for key in model_dict.keys():
        #     if 'mask' in key:
        #         mask_dict[key] = model_dict[key]
        #
        # del model_dict
        # torch.cuda.empty_cache()
        #
        # state_dict = {}
        #
        # for k in model.state_dict().keys():
        #     if k in mask_dict:
        #         mask = mask_dict[k]
        #         values = model.state_dict()[k.replace("_mask","_orig")]
        #         # values[mask==0]= model_pretrained.state_dict()[k][mask==0]
        #         # values[mask == 0] = 0
        #         state_dict[k.replace("_mask","_orig")] = values * mask
        #         state_dict[k] = mask
        #     else:
        #         state_dict[k] = model.state_dict()[k]
        # model.load_state_dict(state_dict)
        # del state_dict
        # torch.cuda.empty_cache()
        ########### zero masking ############################################################
        # ############################################################
        # pruning_model_custom(model.module if hasattr(model, "module") else model, mask_dict,
        #                      'module.' if n_gpu > 1 else '')
        # del mask_dict
        # torch.cuda.empty_cache()
        print('optimizer rewinding')
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if "vil_" in key:
                    lr = 1e-4
                else:
                    lr = base_lr
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
        if args.optim == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=base_lr,
                              eps=args.adam_epsilon,
                              betas=args.adam_betas,
                              correct_bias=args.adam_correct_bias)
        elif args.optim == "RAdam":
            optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
        num_train_optim_steps = (task2num_iters[task] * args.optim_train_epochs // args.grad_acc_steps)
        warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
        if args.lr_scheduler == "warmup_linear":
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
        else:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

        torch.cuda.empty_cache()
        score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu, tb_logger,
                         args.max_val_batches)
        if score > max_score:
            max_score = score
            save_prunned(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu,
                 max_score, is_best=True)
        elif (not args.save_best_only) and ((epoch_id + 1) % args.save_every_num_epochs == 0):
            save_prunned(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu,
                 max_score)

    tb_logger.txt_close()
    print("Best Validation score: %.3f " % (max_score * 100.0))


def evaluate(config, dataloader_val, task_cfg, device, task_id, model, criterion, epoch_id, default_gpu, tb_logger,
             num_batches=-1):
    model.eval()
    for i, batch in enumerate(dataloader_val):
        if i == (num_batches - 1):
            break
        loss, score, batch_size = ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion)
        tb_logger.step_val(epoch_id, float(loss), float(score), task_id, batch_size, "val")
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(dataloader_val)))
            sys.stdout.flush()
    score = tb_logger.showLossVal(task_id)
    model.train()
    return score


if __name__ == "__main__":
    main()