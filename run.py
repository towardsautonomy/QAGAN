import os
import torch
import csv
import src.util as util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering

from src.args import get_args
from src.trainer import Trainer

def main():
    # define parser and arguments
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    util.set_seed(args.seed)

    # load pre-trained model
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        logger = util.get_logger(args.save_dir, 'log_train')
    elif args.do_eval:
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        logger = util.get_logger(args.save_dir, f'log_{split_name}')

    # define trainer
    trainer = Trainer(args, model, tokenizer, logger)

    if args.do_train:
        # train the model
        trainer.train()
    elif args.do_eval:
        # load pretrained model
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)

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