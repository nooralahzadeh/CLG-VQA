import torch
import torch.nn as nn
import numpy as np

from .encoders import BertForVLTasksEncoder, M3PForVLTasks
from .decoder import DecoderModel
from transformers import AutoTokenizer,AutoConfig
from transformers import  XLMRobertaForCausalLM
from transformers.modeling_utils import PreTrainedModel

from .embeddings  import *

class VLencoderdecoderModel(nn.Module):
    def __init__(self, args, config, task_cfg ,task):
        super(VLencoderdecoderModel, self).__init__()
        self.args = args
        self.args.max_seq_length = 7
        self.config = config

        #self.tokenizer = tokenizer
        if "roberta" in args.bert_model:
            config.model = "roberta"
        if config.image_embeddings == "m3p":
            self.VLEncoder = M3PForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
        else:
            self.VLEncoder = BertForVLTasksEncoder.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        tgt_vocab_size = 1576  # the answer embedding
        #self.decoder = DecoderModel(args, config, tgt_vocab_size, self.tokenizer.bos_token_id,self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)

        # initialize word embedding
        self.tgt_embeddings = RobertaTgtEmbeddings(config, tgt_vocab_size, 6)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6, norm=nn.LayerNorm(config.hidden_size))
        self.generator = nn.Linear(config.hidden_size, tgt_vocab_size)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, question, features, spatials, task_id, segment_ids, input_mask, image_mask, tgt_ids, answerEm2tokenEm, mode='train'):

        encoded_layers_t, encoded_layers_v = self.VLEncoder(question, features, spatials, task_id, segment_ids, input_mask, image_mask)
        '''
        src (S, B, E)
        tgt (T, B, E)
        src_mask: :math:`(S, S)`.
        tgt_mask: :math:`(T, T)`.
        
        src_key_padding_mask: :math:`(N, S)`.
        tgt_key_padding_mask: :math:`(N, T)`.
        '''
        src = torch.cat((encoded_layers_v, encoded_layers_t), 1)
        src_padding_mask= torch.cat((image_mask, input_mask), 1)
        src_padding_mask = (src_padding_mask == 0)
        src_mask = torch.zeros((src.size(1), src.size(1)),device=src.device).type(torch.bool)
        #tgt_key_padding_mask = self.create_pad_mask(tgt_ids, self.tokenizer.pad_token_id)

        tgt_input = tgt_ids[:,:-1]
        tgt_padding_mask = (tgt_input == self.tokenizer.pad_token_id)
        #print("tgt_padding_mask",tgt_padding_mask)
        tgt_input_emb = self.tgt_embeddings(tgt_input)

        tgt_sqr_mask = self.generate_square_subsequent_mask(tgt_input_emb)



        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt_input_emb = tgt_input_emb.permute(1, 0, 2)
        #####

        if mode == 'train':
            output = self.decoder(tgt_input_emb, src, tgt_sqr_mask, None, tgt_padding_mask, src_padding_mask) #tgt_key_padding_mask
            print("output", output.size(),output)
            output = self.generator(output)
            print("output",
                  output.transpose(0, 1).contiguous().view(-1, output.size(-1)))


        #     with torch.no_grad():
        #         prediction, _ = self.decoder(encoded_layers_t, encoded_layers_t, mode='sample')
        #
        # elif mode == 'sample':
        #     output, _ = self.decoder(encoded_layers_t, mode='sample')
        #
        #     prediction = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        else:
            raise ValueError
        return output

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


    def generate_square_subsequent_mask(self, tgt):
        mask = (torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class VLRoberta2RobertaModel(PreTrainedModel):
    def __init__(self, args, config, task_cfg ,task):
        super(VLRoberta2RobertaModel, self).__init__()
        self.args = args
        self.args.max_seq_length = 7
        self.config = config

        #self.tokenizer = tokenizer
        if "roberta" in args.bert_model:
            config.model = "roberta"
        if config.image_embeddings == "m3p":
            self.VLEncoder = M3PForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
        else:
            self.VLEncoder = BertForVLTasksPrompt.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        tgt_vocab_size = 1578  # the answer embedding
        decoder_config = AutoConfig.from_pretrained("roberta-base")
        decoder_config.is_decoder = True
        self.decoder = XLMRobertaForCausalLM .from_pretrained("args.bert_model", config=decoder_config)

    def tie_weights(self):
        # for now no weights tying in encoder-decoder
        pass

    def get_encoder(self):
        return self.VLEncoder

    def get_decoder(self):
        return self.decoder


    def get_input_embeddings(self):
        return self.VLEncoder.get_input_embeddings()


    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, question, features, spatials, task_id, segment_ids, input_mask, image_mask, prompt_ids, answerEm2tokenEm, mode='train'):

        encoded_layers_t, encoded_layers_v = self.VLEncoder(question, features, spatials, task_id, segment_ids, input_mask, image_mask)
        encoder_input = torch.cat((encoded_layers_v, encoded_layers_t), 1)
        prompt_masks = torch.ones_like(prompt_ids).to(prompt_ids.device)
        prompt_masks[prompt_ids == self.tokenizer.pad_token_id] = 0


        #####
        if mode == 'train':
            output = self.decoder(encoder_input, encoder_input, prompt_ids, mode='forward')
            with torch.no_grad():
                prediction, _ = self.decoder(encoded_layers_t, encoded_layers_t, mode='sample')

        elif mode == 'sample':
            output, _ = self.decoder(encoded_layers_t, mode='sample')

            prediction = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        else:
            raise ValueError
        return output, prompt_masks, prediction

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs, _ = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)

        return {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

    def _reorder_cache(self, past, beam_idx):
        # as a default encoder-decoder models do not re-order the past.
        # TODO(PVP): might have to be updated, e.g. if GPT2 is to be used as a decoder
        return past
