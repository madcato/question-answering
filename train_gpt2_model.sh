export TRAIN_FILE=train_list.txt
export VAL_FILE=val_list.txt

python3 ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --validation_file=$VAL_FILE
