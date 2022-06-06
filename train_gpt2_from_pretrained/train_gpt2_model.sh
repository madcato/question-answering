# export TRAIN_FILE=train_list.txt
# export VAL_FILE=val_list.txt

export TRAIN_FILE=train_list_tiny.txt

# Train GPT2 model from scratch with custom train files
CUDA_LAUNCH_BLOCKING=1 python3 ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path sshleifer/tiny-gpt2 \
    --train_file $TRAIN_FILE \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --do_train \
    --output_dir output_pretrained \
    --keep_linebreaks False \
    --block_size 64 \
    --learning_rate 2e-2 \
    --num_train_epochs 6000.0
    # --config_overrides "n_embd=768,n_head=8,n_layer=8,n_positions=64"

