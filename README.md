# DQN-FV

Source code and data for the ACL 2021 paper [](A DQN-based Approach to Finding Precise Evidences for Fact Verification).

More information about the FEVER 1.0 shared task can be found on this [https://fever.ai/](website).


## Requirement

- python 3.6.10
- pytorch 1.3.1
- transformers 2.5.1
- prettytable


## Dataset Preparation

The structure of data folder looks like:
```
├── data
│   ├── bert
│   │   └── roberta-large
│   ├── dqn
│   ├── fever
│   ├── glue
│   └── retrieved
```

To replicate the experiments, you need to download these data as following, or diectly obtain them at [](Google Clound).

- `bert`: you can download the roberta pre-trained model from the following wesites and put them into `bert/roberta-large`.
```
wget -O data/bert/roberta-large/pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget -O data/bert/vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget -O data/bert/roberta-large/merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
wget -O data/bert/roberta-large/config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
```
- `fever`: you can download `train.jsonl,shared_task_dev.jsonl,shared_task_test.jsonl` at [https://fever.ai/resources.html](website) and `fever.db` from [https://github.com/thunlp/GEAR](GEAR), and then put them in `fever`.
- `retrieved`: following [https://github.com/thunlp/GEAR](GEAR), we use the document retrieval results from [https://github.com/UKPLab/fever-2018-team-athene](Athene UKP TU Darmstadt) and sentence selection results from [https://github.com/thunlp/GEAR](GEAR). 
- `dqn`: you should first prepare the `retrieved` and then run `sh data_propress.sh` to process data.
- `glue`: you should first prepare the `retrieved` and then run `sh data_process_for_pretrained.sh` to process data.


## Training

Before training, you need to fine-tune the sentence encoding module first. 

### Fine-tune roberta

Run `sh pretrained.sh` first to fine-tune the roberta and then replace `pytorch_model.bin` in the best checkpint with `pytorch_model.bin` in `data/bert/roberta-large`.

You can also directly download our fine-tune version at [](Google Drive).

### Train DQN

Run `sh train.sh` to train our DQN-based model. All checkpoints of our DQN-based model can be found at [](Google Drive).

If you train the model first time, it will spend a long time (about 1 days on our machine) for the sentence encoding module to process the sentences into corressponding semantic representations. Due to the large size, we do not upload the processed-ready data to the cloud. You can directly email `wanhai@mail.sysu.edu.cn` to obtain the data.

Note: the following commands in `train.sh` are to set the version of our DQN-based model. Please choose one before training.
```
## T-T
export DQN_MODE=transformer  # context sub-module
export AGGREGATE=transformer # aggregation sub-module
export ID=TT

## T-A
export DQN_MODE=transformer
export AGGREGATE=attention
export ID=TA

## BiLSTM-T
export DQN_MODE=lstm
export AGGREGATE=transformer
export ID=LT

## BiLSTM-A
export DQN_MODE=lstm
export AGGREGATE=attention
export ID=LA
```

## Testing

Run `sh dev.sh`/`sh test.sh` to evaluate our approach on DEV/TEST set.

After evaluating on TEST, you should submit `test_precise_with/without_post_processing.jsonl` to [https://competitions.codalab.org/competitions/18814#participate](Codalab) to view the blind-test results. 

Note: the foolowing commands in `dev.sh/test.sh` are to set the version of our DQN-based model. Please note that the `CHECKPOINT` in the shell should be kept the same with the version.
```
# context sub-module
export DQN_MODE=transformer
export DQN_MODE=lstm

# aggregation sub-module
export AGGREGATE=transformer
export AGGREGATE=attention
```

## Cite

If you use the code, please cite our paper:
```
@inproceedings{
  title={A DQN-based Approach to Finding Precise Evidences for Fact Verification},
  author={Hai, Wan and Haicheng, Chen and Jianfeng, Du and Weilin, Luo and Rongzhen, Ye},
  booktitle={Proceedings of ACL},
  year={2021}
}
```


## Contact

if you have questions, suggestions and bug reports, please email:
```
wanhai@mail.sysu.edu.cn
```
