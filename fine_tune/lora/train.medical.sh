python train.py \
    --dataset_path ../../data/lora/train.json \
    --model_path ../../data/model/base \
    --base_model ChatGLM2-6B\
    --lora_rank 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 10000 \
    --save_steps 720 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --output_dir adapt \
    --max_seq_length 1024

