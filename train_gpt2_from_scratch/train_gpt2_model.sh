export TRAIN_FILE=train_list.txt
export VAL_FILE=val_list.txt

# Train GPT2 model from scratch with custom train files
CUDA_LAUNCH_BLOCKING=1 python3 ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file $TRAIN_FILE \
    --validation_file=$VAL_FILE \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir output \
    --keep_linebreaks False \
    --block_size 32 \
    --num_train_epochs 32 \
    --config_overrides "n_embd=768,n_head=12,n_layer=12,n_positions=32"
