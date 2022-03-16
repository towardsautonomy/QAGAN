import argparse
import json
from lib2to3.pgen2.tokenize import tokenize
import os
from collections import OrderedDict
import torch
import csv
import src.util as util
from transformers import AdamW
from tensorboardX import SummaryWriter
from textattack.augmentation import EmbeddingAugmenter

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    tokenized_examples["labels"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # label corresponding to the dataset class
        tokenized_examples["labels"].append(dataset_dict['labels'][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def prepare_train_data(dataset_dict, tokenizer):
    # remove questions that are too long
    question_idx_too_long = sorted([i for i in range(len(dataset_dict['question'])) if len(dataset_dict['question'][i]) > 384], reverse=True)
    for idx_to_remove in question_idx_too_long:
        for key in dataset_dict:
            dataset_dict[key].pop(idx_to_remove)
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=129,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["id"] = []
    tokenized_examples["labels"] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # label corresponding to the dataset class
        tokenized_examples["labels"].append(dataset_dict['labels'][sample_index])

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1

            tokenized_examples["end_positions"].append(token_end_index + 1)
            tokenized_examples["start_positions"].append(token_start_index - 1)

            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0] and \
                    context[offset_st : offset_en].lower() != answer['text'][0].lower():
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples

def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples

class Trainer():
    def __init__(self, args, model, tokenizer, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = torch.device(args.device)
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.logger = logger
        self.no_visualization = args.no_visualization
        self.variant = args.variant
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if args.do_train:
            logger.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
            if args.finetune:
                logger.info("Preparing Fine-Tuning Training Data...")
                train_dataset, _ = get_dataset(args, args.finetune_datasets, args.finetune_train_dir, self.tokenizer, 'train')
            else:
                logger.info("Preparing Training Data...")
                train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, self.tokenizer, 'train', args.decimate_dataset, args.upsample_ood)
            if args.finetune:
                logger.info("Preparing Fine-Tuning Validation Data...")
                self.val_dataset, self.val_dict = \
                    get_dataset(args, args.finetune_datasets, args.finetune_val_dir, self.tokenizer, 'val')
            else:
                logger.info("Preparing Validation Data...")
                self.val_dataset, self.val_dict = \
                    get_dataset(args, args.train_datasets, args.val_dir, self.tokenizer, 'val')
            self.train_dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    sampler=RandomSampler(train_dataset))
            self.val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=args.batch_size,
                                    sampler=SequentialSampler(self.val_dataset))

        if args.do_eval:
            args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            split_name = 'test' if 'test' in args.eval_dir else 'validation'
            self.model.to(args.device)
            self.eval_dataset,self.eval_dict = \
                get_dataset(args, args.eval_datasets, args.eval_dir, self.tokenizer, split_name)
            self.eval_dataloader = DataLoader(self.eval_dataset,
                                     batch_size=args.batch_size,
                                     sampler=SequentialSampler(self.eval_dataset))

    def save(self):
        self.model.save_pretrained(self.path)

    def evaluate(self, dataloader, data_dict, 
                       return_preds=False, split='validation'):
        device = self.device

        self.model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch in dataloader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                batch_size = len(input_ids)
                # model inputs
                model_input_dict = {'input_ids': input_ids,
                                    'attention_mask': attention_mask,
                                    'labels': labels}

                outputs = self.model(**model_input_dict)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions( data_dict,
                                                 dataloader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self):
        device = self.device
        self.model.to(device)
        optim = AdamW(self.model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.logger.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(self.train_dataloader.dataset)) as progress_bar:
                for batch in self.train_dataloader:
                    optim.zero_grad()
                    self.model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)

                    # model inputs
                    model_input_dict = {'input_ids': input_ids,
                                        'attention_mask': attention_mask,
                                        'start_positions': start_positions,
                                        'end_positions': end_positions}

                    labels = None
                    if 'use_discriminator' in self.model.config.__dict__.keys():
                        labels = batch['labels'].to(device) \
                                 if self.model.config.__dict__['use_discriminator'] else None
                        model_input_dict['labels'] = labels

                    outputs = self.model(**model_input_dict)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()

                    # update progress bar
                    progress_bar.update(len(input_ids))
                    if hasattr(outputs, 'loss_dict'):
                        loss_dict = outputs.loss_dict.copy()
                        if 'use_discriminator' not in self.model.config.__dict__.keys():
                            progress_bar.set_postfix(epoch=epoch_num, **loss_dict)
                            for k, v in outputs.loss_dict.items():
                                tbx.add_scalar(f'train/{k}', v, global_idx)
                    else:
                        progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                        tbx.add_scalar(f'train/NLL', loss.item(), global_idx)
                        
                    # check if a forward pass through discriminator is required
                    if 'use_discriminator' in self.model.config.__dict__.keys():
                        if self.model.config.__dict__['use_discriminator']:
                            optim.zero_grad()
                            model_input_dict['discriminator'] = True
                            outputs = self.model(**model_input_dict)
                            loss = outputs[0]
                            loss.backward()
                            optim.step()

                            # update progress bar
                            if hasattr(outputs, 'loss_dict'):
                                # update loss_dict
                                for k, v in outputs.loss_dict.items():
                                    loss_dict[k] = v 
                                progress_bar.set_postfix(epoch=epoch_num, **loss_dict)
                                for k, v in loss_dict.items():
                                    tbx.add_scalar(f'train/{k}', v, global_idx)

                    # evaluate
                    if (global_idx % self.eval_every) == 0:
                        self.logger.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(self.val_dataloader, self.val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.logger.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.logger.info(f'Eval {results_str}')
                        if not self.no_visualization:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=self.val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save()
                    global_idx += 1
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name, should_decmiate=False, num_upsample=1):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''

    dataset_sample_fraction = {
        "duorc_augmented": 1,
        "nat_questions_augmented": .3,
        "newsqa_augmented": .3,
        "race_augmented": 1,
        "relation_extraction_augmented": 1,
        "squad_augmented": .3
    }
    for i, dataset in enumerate(datasets):
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        original_dataset_ids = set()
        if "augmented" in dataset:
            try:
                original_dataset_name = dataset.replace("_augmented", "")
                dataset_dict_orginal = util.read_squad(f'{data_dir}/{original_dataset_name}')
                original_dataset_ids = set(dataset_dict_orginal["id"])
            except Exception as e:
                print (e)
                pass

        if should_decmiate:
            dataset_dict_curr = util.downsample_dataset_dir(dataset_dict_curr, dataset_sample_fraction[dataset], original_dataset_ids)
        import copy
        import uuid
        if dataset in ['duorc', 'race', 'relation_extraction', 'duorc_augmented', 'race_augmented', 'relation_extraction_augmented']:
            temp_dataset_dict = copy.deepcopy(dataset_dict_curr)
            for _ in range(num_upsample - 1):
                for key in dataset_dict_curr:
                    if key == 'id':
                        dataset_dict_curr[key].extend([str(uuid.uuid4()) for _ in range(len(temp_dataset_dict[key]))])
                    else:
                        dataset_dict_curr[key].extend(copy.deepcopy(temp_dataset_dict[key]))

        dataset_dict = util.merge(dataset_dict, dataset_dict_curr, i)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict