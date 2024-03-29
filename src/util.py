import json
import random
import os
import logging
import pickle
import string
import re
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict as ddict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return

def visualize(tbx, pred_dict, gold_dict, step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)
    id2index = {curr_id : idx for idx, curr_id in enumerate(gold_dict['id'])}
    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        idx_gold_dict = id2index[id_]
        question = gold_dict['question'][idx_gold_dict]
        context = gold_dict['context'][idx_gold_dict]
        answers = gold_dict['answer'][idx_gold_dict]
        gold = answers['text'][0] if answers else 'N/A'
        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag=f'{split}/{i+1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)


def get_output_dir(base_dir, variant, base_model, name, id_max=100):
    for uid in range(1, id_max):
        output_dir = os.path.join(base_dir, f'{variant}.{base_model}.{name}-{uid:02d}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return output_dir

    raise RuntimeError('Too many output directories created with the same name. \
                       Delete old output directories or use another name.')


def filter_encodings(encodings):
    filter_idx = [idx for idx, val in enumerate(encodings['end_positions'])
                 if not val]
    filter_idx = set(filter_idx)
    encodings_filtered = {key : [] for key in encodings}
    sz = len(encodings['input_ids'])
    for idx in range(sz):
        if idx not in filter_idx:
            for key in encodings:
                encodings_filtered[key].append(encodings[key][idx])
    return encodings_filtered

def merge(encodings, new_encoding, dataset_id):
    dataset_class = [dataset_id] * len(new_encoding['id'])
    new_encoding['labels'] = dataset_class
    if not encodings:
        return new_encoding
    else:
        for key in new_encoding:
            encodings[key] += new_encoding[key]
        return encodings

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    class CustomFormatter(logging.Formatter):
        """ Custom format for the logger
        """

        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format = "[%(asctime)s][%(levelname)s] %(message)s"

        FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: grey + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, datefmt='%m.%d.%y %H:%M:%S')
            return formatter.format(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, f'{name}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(CustomFormatter())

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

class QADataset(Dataset):
    def __init__(self, encodings, train=True):
        self.encodings = encodings
        self.keys = ['input_ids', 'attention_mask', 'labels']
        if train:
            self.keys += ['start_positions', 'end_positions']
        assert(all(key in self.encodings for key in self.keys))

    def __getitem__(self, idx):
        return {key : torch.tensor(self.encodings[key][idx]) for key in self.keys}

    def __len__(self):
        return len(self.encodings['input_ids'])

class QADatasetGen(Dataset):
    def __init__(self, args, 
                       logger, 
                       tokenizer, 
                       dataset_dict, 
                       stride=129,
                       max_length=384,
                       split_name='train'):
        self.tokenizer = tokenizer
        self.dataset_dict = dataset_dict
        self.stride = stride
        self.max_length = max_length
        self.split_name = split_name
        # remove questions that are too long
        question_idx_too_long = sorted([i for i in range(len(self.dataset_dict['question'])) if len(self.dataset_dict['question'][i]) > self.max_length], reverse=True)
        for idx_to_remove in question_idx_too_long:
            for key in self.dataset_dict:
                self.dataset_dict[key].pop(idx_to_remove)

        self.keys = ['input_ids', 'attention_mask', 'labels']
        if split_name == 'train':
            self.keys += ['start_positions', 'end_positions']

    def __getitem__(self, idx):
        # build dataset dictionary for a specific index
        dataset_dict_ = {key : [self.dataset_dict[key][idx]] for key in self.dataset_dict.keys()}
        
        tokenized_examples = self.tokenizer(
                                        dataset_dict_['question'],
                                        dataset_dict_['context'],
                                        truncation="only_second",
                                        stride=self.stride,
                                        max_length=self.max_length,
                                        return_overflowing_tokens=True,
                                        return_offsets_mapping=True,
                                        padding='max_length'
                                    )

        sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["id"] = []
        tokenized_examples["labels"] = []

        # train set
        if self.split_name == 'train':
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            inaccurate = 0
            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answer = dataset_dict_['answer'][sample_index]
                # Start/end character index of the answer in the text.
                start_char = answer['answer_start'][0]
                end_char = start_char + len(answer['text'][0])
                tokenized_examples['id'].append(dataset_dict_['id'][sample_index])
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # label corresponding to the dataset class
                tokenized_examples["labels"].append(dataset_dict_['labels'][sample_index])

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
                    context = dataset_dict_['context'][sample_index]
                    offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
                    offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
                    if context[offset_st : offset_en] != answer['text'][0] and \
                            context[offset_st : offset_en].lower() != answer['text'][0].lower():
                        inaccurate += 1
        else:
            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["id"].append(dataset_dict_["id"][sample_index])
                # label corresponding to the dataset class
                tokenized_examples["labels"].append(dataset_dict_['labels'][sample_index])
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == 1 else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]


        assert(all(key in tokenized_examples for key in self.keys))
        # choose one randomly
        return {key : torch.tensor(tokenized_examples[key][
                            random.choice(range(len(tokenized_examples[key])))
                        ]) for key in self.keys}

    def __len__(self):
        return len(self.dataset_dict['id'])

def write_squad(path, data_dict):
    """
    Write data dict into squad_augmented data.
    Args:
        path: path to write data to
        data_dict: data dict of the same format that is returned by read_squad function

    Returns:

    """

    def construct_answer_entry(answer_dict):
        return [
            {
                "answer_start": answer_dict["answer_start"][i],
                "text": answer_dict["text"][i]
            }
            for i in range(len(answer_dict["answer_start"]))
        ]

    def construct_data_entry(data_dict, record_idx):
        data_entry = {"paragraphs": [
            {
                "context": data_dict["context"][record_idx],
                "context_perplexity": data_dict["context_perplexity"][record_idx],
                "qas": [
                    {
                        "question": data_dict["question"][record_idx],
                        "id": data_dict["id"][record_idx],
                        "answers": construct_answer_entry(data_dict["answer"][record_idx]),
                        "question_perplexity": data_dict["question_perplexity"][record_idx],
                    }
                ]
            }

        ]}
        return data_entry

    path = Path(path)
    squad_dict = {"data": []}

    total_record = len(data_dict['id'])
    for record_idx in range(total_record):
        squad_dict['data'].append(construct_data_entry(data_dict, record_idx))

    with open(path, 'w') as f:
        json.dump(squad_dict, f)

def downsample_dataset_dir(data_dict, sample_fraction, orignal_ids=set()):
    new_data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for i in range(len(data_dict['id'])):
        if data_dict['id'][i] in orignal_ids:
            new_data_dict['question'].append(data_dict['question'][i])
            new_data_dict['context'].append(data_dict['context'][i])
            new_data_dict['id'].append(data_dict['id'][i])
            new_data_dict['answer'].append(data_dict['answer'][i])
        elif random.random() < sample_fraction:

            percentage = 100
            perplexity = 0
            if data_dict['context_perplexity'][i] is not None:
                percentage = data_dict['context_perplexity'][i]['translated_text'] / data_dict['context_perplexity'][i]['original_text']
                perplexity = data_dict['context_perplexity'][i]['translated_text']
            else:
                percentage = data_dict['question_perplexity'][i]['translated_text'] / data_dict['question_perplexity'][i]['original_text']
                perplexity = data_dict['question_perplexity'][i]['translated_text']

            if percentage > 2 and perplexity > 400:
                continue

            new_data_dict['question'].append(data_dict['question'][i])
            new_data_dict['context'].append(data_dict['context'][i])
            new_data_dict['id'].append(data_dict['id'][i])
            new_data_dict['answer'].append(data_dict['answer'][i])
    return new_data_dict


def read_squad(path):
    logging.info('Processing: {}'.format(path))
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': [], 'context_perplexity': [], 'question_perplexity': []}
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            context_perplexity = passage.get('context_perplexity', None)
            for qa in passage['qas']:
                question = qa['question']
                question_perplexity = qa.get('question_perplexity', None)
                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in  qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer)
                        data_dict['context_perplexity'].append(context_perplexity)
                        data_dict['question_perplexity'].append(question_perplexity)
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'context_perplexity': [], 'question_perplexity': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        try:
            data_dict_collapsed['context_perplexity'].append(data_dict['context_perplexity'][ex_ids[0]])
            data_dict_collapsed['question_perplexity'].append(data_dict['question_perplexity'][ex_ids[0]])
        except Exception as e:
            data_dict_collapsed['context_perplexity'].append(None)
            data_dict_collapsed['question_perplexity'].append(None)

        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({'answer_start': [answer['answer_start'] for answer in all_answers],
                                                  'text': [answer['text'] for answer in all_answers]})
    return data_dict_collapsed

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad_augmented answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def convert_tokens(eval_dict, qa_id, y_start_list, y_end_list):
    """Convert predictions to tokens from the context.

    Args:
        eval_dict (dict): Dictionary with eval info for the dataset. This is
            used to perform the mapping from IDs and indices to actual text.
        qa_id (int): List of QA example IDs.
        y_start_list (list): List of start predictions.
        y_end_list (list): List of end predictions.
        no_answer (bool): Questions can have no answer. E.g., SQuAD 2.0.

    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted answer text.
        sub_dict (dict): Dictionary UUIDs -> predicted answer text (submission).
    """
    pred_dict = {}
    sub_dict = {}
    for qid, y_start, y_end in zip(qa_id, y_start_list, y_end_list):
        context = eval_dict[str(qid)]["context"]
        spans = eval_dict[str(qid)]["spans"]
        uuid = eval_dict[str(qid)]["uuid"]
        start_idx = spans[y_start][0]
        end_idx = spans[y_end][1]
        pred_dict[str(qid)] = context[start_idx: end_idx]
        sub_dict[uuid] = context[start_idx: end_idx]
    return pred_dict, sub_dict

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_dicts(gold_dict, pred_dict):
    avna = f1 = em = total = 0
    id2index = {curr_id : idx for idx, curr_id in enumerate(gold_dict['id'])}
    for curr_id in pred_dict:
        total += 1
        index = id2index[curr_id]
        ground_truths = gold_dict['answer'][index]['text']
        prediction = pred_dict[curr_id]
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}
    return eval_dict

def postprocess_qa_predictions(examples, features, predictions,
                               n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = ddict(list)
    for i, feat_id in enumerate(features['id']):
        features_per_example[example_id_to_index[feat_id]].append(i)

    # The dictionaries we have to fill.
    all_predictions = OrderedDict()

    # Let's loop over all the examples!
    for example_index in tqdm(range(len(examples['id']))):
        example = {key : examples[key][example_index] for key in examples}
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            seq_ids = features.sequence_ids(feature_index)
            non_pad_idx = len(seq_ids) - 1
            while not seq_ids[non_pad_idx]:
                non_pad_idx -= 1
            start_logits = start_logits[:non_pad_idx]
            end_logits = end_logits[:non_pad_idx]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features["offset_mapping"][feature_index]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features.get("token_is_max_context", None)
            if token_is_max_context:
                token_is_max_context = token_is_max_context[feature_index]


            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either = 0 or > max_answer_length.
                    if end_index <= start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "start_index": start_index,
                            "end_index": end_index,
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred['offsets']
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # need to find the best non-empty prediction.
        i = 0
        while i < len(predictions):
            if predictions[i]['text'] != '':
                break
            i += 1
        if i == len(predictions):
            import pdb; pdb.set_trace();

        best_non_null_pred = predictions[i]
        all_predictions[example["id"]] = best_non_null_pred["text"]

    return all_predictions

# All methods below this line are from the official SQuAD 2.0 eval script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1