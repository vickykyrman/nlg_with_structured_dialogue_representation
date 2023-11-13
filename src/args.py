
import argparse
import os
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

def parse():

    parser = argparse.ArgumentParser()

    #arguments to be called from terminal
    #the first argument in each row (eg. --mode) is the argument name. The values that the argument can take are given in "help"
    
    #TO EXECUTE model_main.py
    parser.add_argument("--mode", type=str, default='evaluate', help='[train] ---> for training, [evaluate] ---> for evaluating]')
    parser.add_argument('--quality', type=str, default=None, help='The qualitative scenario')
    parser.add_argument('--quantity', type=str, default=None, help='The quantitative scenario')
    parser.add_argument('--perspective', type=bool, default=None, help='Add the structured perspective information')


    #NO ARGUMENTS NEEDED TO EXECUTE data_main.py

    #DATA
    parser.add_argument("--original-path", type=str, default='../data/original.csv')
    parser.add_argument("--mini-original-path", type=str, default='../data/mini/mini_original.csv')
    parser.add_argument("--dial-act-classifier", type=str, default='./data_utils/perspective/classifier.pt')

    #EVALUATION
    parser.add_argument("--evaluation-path", type=str, default='../evaluation/automatic')
                        
    #MODEL
    parser.add_argument("--models-path", type=str, default='../finetuned_models')
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=75,
        help="The maximum total sequence length for target/gold text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val-max-target-length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num-warmup-steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--overwrite-cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="max length"
    )
    parser.add_argument(
        "--pad-to-max-length", type=bool, default=True, help="do pading"
    )
    parser.add_argument(
        "--ignore-pad-token-for-loss", type=bool, default=True, help="do pading"
    )
    parser.add_argument(
        "--logging-steps", type=int, default=500, help="do pading"
    )
    parser.add_argument(
        "--save-steps", type=int, default=None, help="do pading"
    )
    parser.add_argument(
        "--save-every-checkpoint", action="store_false"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="max_grad_norm"
    )
    parser.add_argument(
        "--no-kb", action="store_true"
    )

    return parser.parse_args()

#we run the parse function
ARGS = parse()

#if we're running this individual file
if __name__ == "__main__":
    print('Running args.py')