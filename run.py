import os
import torch
import csv
import src.util as util
import pprint
import json

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertModel, DistilBertConfig

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer, BertTokenizerFast
from transformers import BertModel, BertConfig

from src.args import get_args
from src.model import QAGAN, QAGANConfig
from src.trainer import Trainer

def get_available_devices(force_gpu_ids=[]):
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available() and len(force_gpu_ids) != 0:
        gpu_ids = force_gpu_ids
        device = f'cuda:{gpu_ids[0]}'
        torch.cuda.set_device(torch.device(device))
    elif torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = f'cuda:{gpu_ids[0]}'
        torch.cuda.set_device(torch.device(device))
    else:
        device = 'cpu'

    return device, gpu_ids

def main():
    # define parser and arguments
    args = get_args()
    # get device info
    args.device, args.gpu_ids = get_available_devices(force_gpu_ids=[0])
    args.batch_size*= max(1, len(args.gpu_ids))
    # set random seed
    util.set_seed(args.seed)

    # load pre-trained base model
    model, tokenizer = None, None
    if args.variant == 'baseline':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    else:
        config = BertConfig()
        config.output_hidden_states = False
        config.output_attentions = False
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        backbone = BertModel.from_pretrained("bert-base-uncased", config=config)

        if args.variant == 'baseline-cond':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer,
                                  prediction_head='conditional_linear',
                                  num_datasets=args.num_datasets)
        elif args.variant == 'baseline-cond-att':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer,
                                  prediction_head='conditional_attention',
                                  num_datasets=args.num_datasets)
        elif args.variant == 'qagan':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls=True,
                                  num_datasets=args.num_datasets)
        elif args.variant == 'qagan-hidden':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_hidden_layers=True,
                                  num_datasets=args.num_datasets)
        elif args.variant == 'qagan-cond':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls=True,
                                  prediction_head='conditional_linear',
                                  num_datasets=args.num_datasets)
        elif args.variant == 'qagan-cond-kld':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls=True,
                                  prediction_head='conditional_linear',
                                  constrain_hidden_repr=True,
                                  num_datasets=args.num_datasets)
        elif args.variant == 'qagan-cond-att':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls=True,
                                  prediction_head='conditional_attention',
                                  num_datasets=args.num_datasets)
        elif args.variant == 'qagan-cond-tfm':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls=True,
                                  prediction_head='conditional_transformers',
                                  num_datasets=args.num_datasets)
        else:
            raise ValueError
        # define model
        model = QAGAN(config=qconfig)


    # load pre-trained model
    if args.finetune or args.load_pretrained:
        assert args.pretrained_model != 'none', 'Must specify a pretrained model to load'
        assert args.pretrained_model != '', 'Must specify a pretrained model to load'
        if 'qagan' in args.variant:
            qconfig['anneal'] = False
            qconfig['fake_discriminator_warmup_steps'] = 0
            model = QAGAN(config=qconfig).from_pretrained(args.pretrained_model)
        else:
            model = BertForQuestionAnswering.from_pretrained(args.pretrained_model)
            
    # mode of operation
    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        args.output_dir = util.get_output_dir(args.output_dir, args.variant, args.run_name)
        logger = util.get_logger(args.output_dir, 'log_train')
    elif args.do_eval:
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        logger = util.get_logger(args.output_dir, f'log_{split_name}')

    if args.do_train:
        # define trainer
        trainer = Trainer(args, model, tokenizer, logger)
        # train the model
        trainer.train()
    elif args.do_eval:
        # load pretrained model
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint')
        if args.variant == 'baseline':
            model = BertForQuestionAnswering.from_pretrained(checkpoint_path)
        else:
            model = QAGAN(config=qconfig).from_pretrained(checkpoint_path)
        # define trainer
        trainer = Trainer(args, model, tokenizer, logger)
        # run evaluation
        eval_preds, eval_scores = trainer.evaluate( dataloader=trainer.eval_dataloader,
                                                    data_dict=trainer.eval_dict,
                                                    return_preds=True,
                                                    split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        logger.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.output_dir, split_name + '_' + args.sub_file)
        logger.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])

if __name__ == '__main__':
    main()