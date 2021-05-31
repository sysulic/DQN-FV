# DQN-FV

Source code and data for the ACL 2021 paper [A DQN-based Approach to Finding Precise Evidences for Fact Verification]().

More information about the FEVER 1.0 shared task can be found on this [website](https://fever.ai/).


## Requirement

- python 3.6.10
- pytorch 1.3.1
- transformers 2.5.1
- prettytable


## Dataset Preparation

The structure of the data folder is as follows:
```
├── data
│   ├── bert
│   │   └── roberta-large
│   ├── dqn
│   ├── fever
│   ├── glue
│   └── retrieved
```

To replicate the experiments, you need to download these data as follows, or directly obtain them at [Google Drive](https://drive.google.com/drive/folders/1armZDd2fch8RFH09rfswAIwbdsx_c2QW?usp=sharing).

Note: due to the large size, you should run the following command to download `fever.db` alone and put it into `fever`:
```
# Download the fever database
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db
```

- `bert`: you can download the Roberta pre-trained model with the following commands and put them into `bert/roberta-large`.
```
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
```
- `fever`: you can download `train.jsonl,shared_task_dev.jsonl,shared_task_test.jsonl` from [website](https://fever.ai/resources.html) and `fever.db` from [GEAR](https://github.com/thunlp/GEAR), and then put them in `fever`.
- `retrieved`: following [GEAR](https://github.com/thunlp/GEAR), we use the document retrieval results from [Athene UKP TU Darmstadt](https://github.com/UKPLab/fever-2018-team-athene) and sentence selection results from [GEAR](https://github.com/thunlp/GEAR). 
- `dqn`: you should first prepare data in `retrieved` and then run `sh data_propress.sh` to obtain data in `dqn`.
- `glue`: you should first prepare data in `retrieved` and then run `sh data_process_for_pretrained.sh` to obtain data in `glue`.


## Training

Before training, you need to fine-tune the sentence encoding module (i.e., Roberta) first. 

### Fine-tune Roberta

Run `sh pretrained.sh` first to fine-tune the Roberta and then replace `pytorch_model.bin` in `data/bert/roberta-large` with `pytorch_model.bin` in the best checkpoint.

You can also directly download our fine-tune version at [Google Drive](https://drive.google.com/drive/folders/1armZDd2fch8RFH09rfswAIwbdsx_c2QW?usp=sharing).

### Train DQN

Run `sh train.sh` to train our DQN-based model. All checkpoints of our DQN-based model can be found at [Google Drive](https://drive.google.com/drive/folders/1armZDd2fch8RFH09rfswAIwbdsx_c2QW?usp=sharing).

If you train the model at first, it may spend a long time (about 1 day in our machine) for the sentence encoding module to process the sentences into corresponding semantic representations. Due to the large size, we do not upload the processed-ready data to the cloud. You can directly email `wanhai@mail.sysu.edu.cn` to obtain the data.

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

After evaluating on TEST, you should submit `test_precise_with/without_post_processing.jsonl` to [CodaLab](https://competitions.codalab.org/competitions/18814#participate) to view the blind-test results. 

Note: the following commands in `dev.sh/test.sh` are to set the version of our DQN-based model. Please note that the `CHECKPOINT` in the script should be kept the same as the version.
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
