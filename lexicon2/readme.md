# Lexicon2 

Create a language model using as input a English language dictionary.

Use Huggingface train sripts

## Info
- [Code from Huggingface](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation)

## Install

1. Install requirements
```sh
pip3 install -r requirements.txt
```
2. Download data
```sh
python3 download.py
```
3. Prepare data
```sh
python3 prepare.py
```

## Run

```sh
python3 run_translation.py \
    --model_name_or_path facebook/blenderbot-400M-distill \
    --do_train \
    --do_eval \
    --source_lang input \
    --target_lang output \
    --train_file train.json \
    --validation_file validation.json \
    --output_dir ./tst-translation \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --predict_with_generate
```