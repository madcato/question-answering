# Train GPT2 model from scratch with custom train files
CUDA_LAUNCH_BLOCKING=1 python3 ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path sshleifer/tiny-gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --output_dir output_pretrained_wiki