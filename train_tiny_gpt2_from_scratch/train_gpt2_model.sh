export TRAIN_FILE=train_list_tiny.txt

# Train GPT2 model from scratch with custom train files
CUDA_LAUNCH_BLOCKING=1 python3 ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file $TRAIN_FILE \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --do_train \
    --output_dir output_tiny \
    --keep_linebreaks False \
    --block_size 64 \
    --num_train_epochs 360 \
    --config_overrides "n_embd=768,n_head=8,n_layer=8,n_positions=64"