import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, choices=['baseline', 
                                                        'baseline-cond', 
                                                        'baseline-cond-att',
                                                        'qagan',
                                                        'qagan-hidden',
                                                        'qagan-cond',
                                                        'qagan-cond-kld',
                                                        'qagan-cond-att',
                                                        'qagan-cond-tfm'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--num-datasets', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='output/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--base-model', type=str, default='distil-bert', choices=['bert',
                                                                                  'distil-bert'])
    parser.add_argument('--base-model-id', type=str, default='distilbert-base-uncased', 
                            choices=['distilbert-base-uncased',
                                     'bert-base-uncased'])
    parser.add_argument('--load-pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained-model', type=str, default='none')
    parser.add_argument('--train-datasets', type=str, default='HotpotQA,NaturalQuestions,NewsQA,SearchQA,SQuAD,TriviaQA')
    parser.add_argument('--run-name', type=str, default='default')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--finetune-train-dir', type=str, default='datasets/oodomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--finetune-val-dir', type=str, default='datasets/oodomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_val')
    parser.add_argument('--eval-datasets', type=str, default='BioASQ')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--no-visualization', action='store_true')
    parser.add_argument('--ram-caching', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)

    args = parser.parse_args()
    return args