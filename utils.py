import os
import random
import json
import csv
import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import itertools as it

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, BertTokenizer, AutoModelWithLMHead, AutoConfig
import pytorch_lightning as pl

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings('ignore')

import pdb

class BertNerDataset(Dataset):
    def __init__(self, tokenizer, data_path="./data/train-ner.json", type_path="train", doc_size=512, batch_size=1):
        super(BertNerDataset,).__init__()
        
        self.tokenizer = tokenizer
        
        self.data = []
        
        self.ner_labels = {'O': 0}
        with open(data_path,'rt') as f:
            json_data = json.load(f)
            for set_id,text,targets in tqdm(json_data, desc=f"Loading {type_path} data"):
                iob, codes = self.gen_iob(text,targets)
                
                if len(codes) > doc_size: continue  # remove data that is too large
                    
                tokens = torch.zeros(doc_size, dtype=torch.int64)
                labels = torch.zeros(doc_size, dtype=torch.int64)
                
                tokens[0] = codes[0]
                labels[0] = 0
                
                for i,(token,label) in enumerate(iob):
                    if not label in self.ner_labels:
                        self.ner_labels[label] = max(self.ner_labels.values())+1
                    tokens[i+1] = codes[i+1]
                    labels[i+1] = self.ner_labels[label]
                
                tokens[len(codes)-1] = codes[-1]
                labels[len(codes)-1] = 0
                
                if 'B-IND' in {l for t,l in iob}:  # if there is at least one match
                    self.data.append([set_id,text,targets,iob,tokens,labels])

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text.lower(), add_special_tokens=True)
        codes = self.tokenizer.encode(text.lower(), add_special_tokens=True)

        return tokens, codes

    def get_token_text_locs(self, text, tokens):
        tok_i = 0
        tok_char_i = 0
        char_toks = []
        for i,c in enumerate(text.lower()):            
            if tok_char_i > len(tokens[tok_i])-1:
                tok_i += 1
                tok_char_i = 0
    #         For Roberta
    #         while True:
    #             if tok_i > len(tokens)-1:
    #                 break
    #             if tokens[tok_i] in {'Ċ', 'Ġ'}:
    #                 tok_i += 1
    #                 tok_char_i = 0
    #             else:
    #                 break

            if tok_i > len(tokens)-1:
                break

            if tokens[tok_i][0:2] == '##' and tok_char_i == 0:
                tok_char_i = 2
    #         if tokens[tok_i][0] == 'Ġ' and tok_char_i == 0:
    #             tok_char_i = 1

            if tokens[tok_i][tok_char_i] == c:
                tok_char_i += 1

            char_toks.append(tok_i)

        return char_toks
    
    def remove_overlaps(self, locs):
        remove = set()
        for (b1,e1),(b2,e2) in it.permutations(locs,2):
            if b2>=b1 and e2<=e1:
                if (e2-b2)<(e1-b1):
                    remove.add((b2,e2))

        return [(b,e) for b,e in locs if not (b,e) in remove]

    def get_text_locs(self, s, text):
        s = s.lower()
        text = text.lower()
        l = len(s)
        matches = []
        for i in range(len(text)):
            if text[i:i+l] == s:
                matches.append((i,i+l))
        return matches

    def gen_iob(self, text, targets):
        tokens, codes = self.tokenize(text)
        char_toks = self.get_token_text_locs(text, tokens)

        matches = []
        for s in targets:
            matches.append(self.get_text_locs(s,text))
        matches = sorted([l for m in matches for l in m])
        matches = self.remove_overlaps(matches)

        iob = [[t,"O"] for t in tokens]
        match_toks = [i for b,e in matches for i in range(char_toks[b],char_toks[e-1]+1)]
        for l in match_toks:
            iob[l][1] = "I-IND"
        match_beginnings = [char_toks[b] for b,e in matches]
        for l in match_beginnings:
            iob[l][1] = "B-IND"

        for i,(t,l) in enumerate(iob):
#             if t[:2] == '##':
#                 l = '[PAD]'
            iob[i] = (t,l)

        return iob, codes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        set_id, text, targets, iob, tokens, labels = self.data[index]
        
        # make attention mask
        attention_mask = torch.ones(tokens.shape, dtype=torch.torch.float32)
        end = True
        for i in range(tokens.shape[0]-1,-1,-1):
            if end:
                attention_mask[i] = 0
            if tokens[i] > 0:
                end = False

        attention_mask[0] = 0
        
        return {
            "tokens": tokens.squeeze(),
            "labels": labels.squeeze(),
            "attention_mask": attention_mask.squeeze()
        }
    
    
class BertNerSystem(pl.LightningModule):

    def __init__(self, hparams, user_tokens=['<newline>', '<bullet>']):
        super(BertNerSystem, self).__init__()
        self.hparams = hparams
        self.hparams.model_type = self.hparams.model_type.lower()
        tokenizer = BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            never_split=user_tokens,
            do_lower_case=self.hparams.do_lower_case,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        
        config = AutoConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
            output_past = not self.hparams.do_train,
            num_labels = self.hparams.num_labels,
        )
        model = BertForTokenClassification.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=config,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        self.config, self.tokenizer, self.model = config, tokenizer, model
        self.loss = []  # for keeping track of average loss
        self.metrics = {}
        
        self.vocab = {v:k for k,v in self.tokenizer.get_vocab().items()}
    
    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        try: avg_loss =  sum(self.loss)/len(self.loss)
        except: avg_loss = -1
        self.loss = []
        tqdm_dict = {"loss": "{:.3g}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
#         tqdm_dict = {"loss": "{:.3g}".format(avg_loss)}
        return tqdm_dict

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.doc_max_seq_length),
            ),
        )
        
    def forward(self, input_ids, labels=None, attention_mask=None):
        # input_ids (torch.LongTensor of shape (batch_size, sequence_length)) – Indices of input sequence tokens in the vocabulary. Indices can be obtained using transformers.RobertaTokenizer.
        # attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional, defaults to None) – Mask to avoid performing attention on padding token indices. 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        # token_type_ids (torch.LongTensor of shape (batch_size, sequence_length), optional, defaults to None) – Segment token indices to indicate first and second portions of the inputs. 0 corresponds to sentence A, 1 corresponds to sentence B
        # position_ids (torch.LongTensor of shape (batch_size, sequence_length), optional, defaults to None) – Indices of positions of each input sequence tokens in the position embeddings. Selected in the range [0, config.max_position_embeddings - 1].
        # head_mask (torch.FloatTensor of shape (num_heads,) or (num_layers, num_heads), optional, defaults to None) – Mask to nullify selected heads of the self-attention modules. 1 indicates the head is not masked, 0 indicates the head is masked.
        # inputs_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional, defaults to None) – Optionally, instead of passing input_ids you can choose to directly pass an embedded representation.
        # labels (torch.LongTensor of shape (batch_size, sequence_length), optional, defaults to None) – Labels for computing the token classification loss. Indices should be in [0, ..., config.num_labels - 1].
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=labels,
        )
    
    def _step(self, batch):
        outputs = self.forward(
            input_ids=batch["tokens"].clone(),
            labels=batch["labels"].clone(),
            attention_mask=batch["attention_mask"].clone(),
        )
        
        return outputs[0]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.loss.append(loss.item())  # for keeping track of average loss

#         tensorboard_logs = {"Training/Loss": loss, "Learning rate": self.lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss}  # "log": tensorboard_logs

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.loss.append(loss.item())  # for keeping track of average loss

        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         tensorboard_logs = {"Validation/Loss": avg_loss}
        return {"avg_val_loss": avg_loss}  #  "log": tensorboard_logs

    def test_step(self, batch, batch_idx):
        output, = self.forward(
            input_ids=batch["tokens"].clone(),
            attention_mask=batch["attention_mask"].clone(),
        )
        predictions = torch.argmax(output, dim=2)
        predictions = predictions.squeeze(dim=0).tolist()
#         return {"val_loss": loss.item(), "preds": preds[0], "target": target[0]}
        return batch, predictions

    def test_end(self, outputs):
        exact_subtoken, partial_subtoken = TestMetrics.test_all(outputs, self.vocab, f=TestMetrics.test_predictions_subtoken, f_kwargs={'score_map': {'exact': 2, 'partial': 1, False: 0}})
        exact_token, partial_token = TestMetrics.test_all(outputs, self.vocab, f=TestMetrics.test_predictions_token, f_kwargs={'score_map': {'exact': 2, 'partial': 1, False: 0}})
        exact_nonpositional, partial_nonpositional = TestMetrics.test_all(outputs, self.vocab, f=TestMetrics.test_predictions_nonpositional, f_kwargs={'threshold': 0.33})
        self.metrics = {
            'exact_subtoken': exact_subtoken, 
            'partial_subtoken': partial_subtoken, 
            'exact_token': exact_token, 
            'partial_token': partial_token, 
            'exact_nonpositional': exact_nonpositional, 
            'partial_nonpositional': partial_nonpositional
        }
        return {k:v['f1'] for k,v in self.metrics.items()}

    def test_epoch_end(self, outputs):
        return

    def train_dataloader(self):
        self.train_dataset = BertNerDataset(
            self.tokenizer, data_path=self.hparams.train_data_path, type_path="train", \
            doc_size=self.hparams.doc_max_seq_length, batch_size=self.hparams.train_batch_size
        )
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, shuffle=self.hparams.shuffle_training_data, drop_last=True)

#         t_total = (
#             (len(dataloader.dataset) * self.hparams.num_train_epochs)
#             // (self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps)
#         )
#         warmup_steps = self.hparams.warmup_steps*self.hparams.train_batch_size
#         scheduler = get_linear_schedule_with_warmup(
#             self.opt, num_warmup_steps=warmup_steps, num_training_steps=t_total
#         )
        
        schedule_freq = self.hparams.schedule_every_n_steps // (self.hparams.gradient_accumulation_steps * self.hparams.train_batch_size)
    
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=schedule_freq,
            gamma=self.hparams.lr_decay
        )

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.opt,
#             mode='min',
#             factor=0.1,
#             patience=2,
#             verbose=True,
#             threshold=1e-4,
#             threshold_mode='rel',
#             cooldown=0,
#             min_lr=0,
#             eps=1e-08
#         )

        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        self.val_dataset = BertNerDataset(
            self.tokenizer, data_path=self.hparams.val_data_path, type_path="val", \
            doc_size=self.hparams.doc_max_seq_length, batch_size=self.hparams.eval_batch_size
        )
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        self.test_dataset = BertNerDataset(
            self.tokenizer, data_path=self.hparams.test_data_path, type_path="test", \
            doc_size=self.hparams.doc_max_seq_length, batch_size=self.hparams.test_batch_size
        )
        return DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False, drop_last=True)

class TensorBoardCallback(pl.Callback):
    writer = None
    step_count = 0
    
    def on_test_start(self, trainer, pl_module):
        if not self.writer:
            self.writer = SummaryWriter(log_dir=pl_module.hparams.log_dir, max_queue=10, flush_secs=120)
        
    def on_sanity_check_start(self, trainer, pl_module):
        self.writer = SummaryWriter(log_dir=pl_module.hparams.log_dir, max_queue=10, flush_secs=120)
        self.weights_keys = [f'encoder.layer.{i}.output.LayerNorm.weight' for i in range(24)]
        self.biases_keys = [f'encoder.layer.{i}.output.LayerNorm.bias' for i in range(24)]
        self.val_losses = []
        self.step_count = 0
    
    def on_batch_end(self, trainer, pl_module):
        for i in range(pl_module.hparams.train_batch_size):
            self.step_count += 1
            if self.step_count % pl_module.hparams.hist_log_freq == 0:
                self.writer.add_scalar("Training/Learning-rate", pl_module.lr_scheduler.get_last_lr()[-1], self.step_count)  # log learning rate
                # log gradients, not sure how to do this...

                # log weights and biases
                state = dict(pl_module.model.bert.state_dict())
                for i,k in enumerate(self.weights_keys):
                    v = state[k]
                    title = f'Weights/Layer-{i}'
                    self.writer.add_histogram(title, v, self.step_count)
                for i,k in enumerate(self.biases_keys):
                    v = state[k]
                    title = f'Biases/Layer-{i}'
                    self.writer.add_histogram(title, v, self.step_count)
                    
        loss = trainer.running_loss.last()  # log loss
        if loss:
            self.writer.add_scalar("Training/Loss", loss.item(), self.step_count)  # log training loss
        
#         self.writer.flush()
        
    def on_validation_start(self, trainer, pl_module):
        self.val_losses = []
        
    def on_validation_batch_end(self, trainer, pl_module):
        loss = trainer.running_loss.last()
        if loss:
            self.val_losses.append(loss.item())
        
    def on_validation_end(self, trainer, pl_module):
        if len(self.val_losses):
            avg_loss = sum(self.val_losses)/len(self.val_losses)
            self.writer.add_scalar("Validation/Loss", avg_loss, self.step_count)  # log validation loss
#             self.writer.flush()

    def on_test_end(self, trainer, pl_module):
        for metric_type, metric_data in pl_module.metrics.items():
            for metric_name, value in metric_data.items():
                self.writer.add_scalar(f"Test/{metric_type}-{metric_name}", value, self.step_count)  # log validation loss
        #  self.writer.flush()
    
class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info(f"{key} = {metrics[key]:.3g}\n")
                        writer.write(f"{key} = {metrics[key]:.3g}\n")

class EpochCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # checkpoint
        Path(pl_module.hparams.output_dir).mkdir(parents=True, exist_ok=True)
        fn = f"{pl_module.hparams.output_dir}/checkpoint-step-{trainer.global_step}.ckpt"
        trainer.save_checkpoint(fn)

        fp = f"{pl_module.hparams.output_dir}/checkpoint-{trainer.global_step}"
        Path(fp).mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(fp)
        pl_module.tokenizer.save_pretrained(fp)
        pl_module.config.save_pretrained(fp)
            
        if pl_module.is_logger():
            logger.info(f"Model checkpointed at step {trainer.global_step}")
                        
class CheckpointCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % (pl_module.hparams.checkpoint_every_n_steps//pl_module.hparams.train_batch_size) == 0:
            # checkpoint
            Path(pl_module.hparams.output_dir).mkdir(parents=True, exist_ok=True)
            fn = f"{pl_module.hparams.output_dir}/checkpoint-step-{trainer.global_step}.ckpt"
            trainer.save_checkpoint(fn)
            
            fp = f"{pl_module.hparams.output_dir}/checkpoint-{trainer.global_step}"
            Path(fp).mkdir(parents=True, exist_ok=True)
            pl_module.model.save_pretrained(fp)
            pl_module.tokenizer.save_pretrained(fp)
            pl_module.config.save_pretrained(fp)
            
            if pl_module.is_logger():
                logger.info(f"Model checkpointed at step {trainer.global_step}")
        
class DebugCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        from gpu_profile import gpu_profile
        import sys
        sys.settrace(gpu_profile)

        
        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Dict2Obj(dict):
    """
    Example:
    m = Dict2Obj({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        try: return self.__dict__[attr]
        except KeyError: raise AttributeError(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict2Obj, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict2Obj, self).__delitem__(key)
        del self.__dict__[key]


def get_trainer(pl_module, args):
    # init model
    set_seed(args)

    t = datetime.now()
    prefix = t.strftime('%d%b%Y-%H:%M:%S')

    # Create output dir
    if args.do_train:  #  os.path.exists(args.output_dir) and os.listdir(args.output_dir) and
        args.output_dir = os.path.join(args.output_dir, f"{prefix}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log dir
    args.log_dir = os.path.join(args.log_dir, f"{prefix}")
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    args.gpus = list(range(args.n_gpu))

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        auto_select_gpus=False,  # True might cause memory to run out
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
#         checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), EpochCallback(), TensorBoardCallback()],
        reload_dataloaders_every_epoch=False,
        train_percent_check=args.train_percent_check,
        val_percent_check=args.val_percent_check,
        test_percent_check=args.test_percent_check,
    )

    if hasattr(args, 'profile_gpu_memory'):
        if args.profile_gpu_memory:
            train_params['callbacks'].append(DebugCallback())
    if hasattr(args, 'checkpoint_every_n_steps'):
        if args.checkpoint_every_n_steps > 0:
            train_params['callbacks'].append(CheckpointCallback())
    if hasattr(args, 'val_check_interval'):
        if args.val_check_interval > 0:
            train_params['val_check_interval'] = args.val_check_interval

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "dp"
    else:
        train_params["distributed_backend"] = None

    trainer = pl.Trainer(**train_params)
    
    return trainer
        
def generic_train(trainer, pl_module, args):
    if args.do_train:
        trainer.fit(pl_module)

    return trainer
