export TRAIN_FILE=train_seq2seq_list.csv
export VAL_FILE=val_seq2seq_list.csv

export CUDA_VISIBLE_DEVICES=""

# Train GPT2 model from scratch with custom train files
CUDA_LAUNCH_BLOCKING=1 python3 ./transformers/examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --text_column question \
    --summary_column answer \
    --source_prefix "summarize: " \
    --output_dir output_seq2seq \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --max_source_length 128
