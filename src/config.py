#coding: utf8

def set_com_args(parser) -> None:
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='./data/dqn/',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str
    )
    #parser.add_argument(
    #    "--checkpoints",
    #    default=None,
    #    nargs='+',
    #    type=str
    #)
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--dev_true_file",
        default="data/fever/shared_task_dev.jsonl",
        type=str,
    )
    parser.add_argument(
        "--test_true_file",
        default="data/fever/shared_task_test.jsonl",
        type=str,
    )
    parser.add_argument(
        "--max_evi_size",
        default=5,
        type=int,
        help="The maximum evidence size"
    )
    
    # Other parameters
    parser.add_argument(
        "--num_labels",
        default=3,
        type=int
    )
    #parser.add_argument(
    #    "--max_sent_length",
    #    default=64,
    #    type=int,
    #    help="The maximum length of each sentence"
    #)
    #parser.add_argument(
    #    "--max_seq_length",
    #    default=500,
    #    type=int,
    #    help="The maximum total input sequence length after tokenization. Sequences longer "
    #    "than this will be truncated, sequences shorter will be padded.",
    #)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    #parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        #"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
        "--do_lower_case", type=int, choices=[0, 1], help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=12, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=12, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=100.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


def set_dqn_args(parser) -> None:
    parser.add_argument('--dqn_type', default='ddqn', choices=['dqn', 'ddqn'])
    parser.add_argument('--dqn_mode', default='lstm', choices=['bert', 'lstm', 'transformer', 'ggnn'])
    parser.add_argument('--aggregate', default='transformer', type=str, choices=['transformer', 'attention'])
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    # replay memory
    parser.add_argument('--capacity', default=10000, type=int)
    # discount factor
    parser.add_argument('--eps_gamma', default=0.95, type=float)
    # epsilon greedy
    parser.add_argument('--eps_start', default=0.9, type=float)
    parser.add_argument('--eps_end', default=0.05, type=float)
    parser.add_argument('--eps_decay', default=1000, type=float)

    parser.add_argument('--target_update', default=10, type=int)
    parser.add_argument('--tau', default=1., type=float)

    parser.add_argument('--mem', default='label_priority', choices=['random', 'priority', 'label_random', 'label_priority'])
    parser.add_argument('--proportion', default=[1, 1, 1], nargs=3, type=float)
    parser.add_argument('--pred_thred', default=0.1, type=float)

def set_bert_args(parser) -> None:
    #from dqn.bert_dqn import MODEL_CLASSES
    from data.load_data import MODEL_CLASSES
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default='./data/bert/bert-base-uncased',
        type=str,
        help="BERT retrained model"
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--task_name",
        default='mnli',
        type=str,
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )

    #parser.add_argument(
    #    "--gradient_accumulation_steps",
    #    type=int,
    #    default=1,
    #    help="Number of updates steps to accumulate before performing a backward/update pass.",
    #)
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )

