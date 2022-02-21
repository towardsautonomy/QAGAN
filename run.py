import os
import torch
import csv
import src.util as util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertModel, DistilBertConfig

from src.args import get_args
from src.model import QAGAN, QAGANConfig
from src.trainer import Trainer

def main():
    # define parser and arguments
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    util.set_seed(args.seed)

    # load pre-trained base model
    model, tokenizer = None, None
    if args.variant == 'baseline-v0':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    elif 'qagan' in args.variant:
        config = DistilBertConfig()
        config.output_hidden_states = False
        config.output_attentions = False
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        backbone = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        if args.variant == 'baseline-v1':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer)
        elif args.variant == 'qagan-v0':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls=True)
        elif args.variant == 'qagan-v1':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_cls_sep=True)
        elif args.variant == 'qagan-v2':
            qconfig = QAGANConfig(backbone=backbone, 
                                  tokenizer=tokenizer, 
                                  use_discriminator=True,
                                  discriminate_hidden_layers=True)
        else:
            raise ValueError

        # get the QAGAN model
        if args.load_pretrained:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
            model = QAGAN(config=qconfig).from_pretrained(checkpoint_path)
        else:
            model = QAGAN(config=qconfig)
    else:
        raise ValueError

    # mode of operation
    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.variant, args.run_name)
        logger = util.get_logger(args.save_dir, 'log_train')
    elif args.do_eval:
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        logger = util.get_logger(args.save_dir, f'log_{split_name}')

    if args.do_train:
        # define trainer
        trainer = Trainer(args, model, tokenizer, logger)
        # train the model
        trainer.train()
    elif args.do_eval:
        # load pretrained model
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        if args.variant == 'baseline':
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
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
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        logger.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])

if __name__ == '__main__':
    main()