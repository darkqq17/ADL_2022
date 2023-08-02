python3 train_MC.py \
  --model_name_or_path ./MC_save_pretrained \
  --cache_dir ./ckpt_MC \
  --output_dir ./ckpt_MC \
  --context_file $1 \
  --test_file $2 \
  --output_file ./selection_pred.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 4 \

python3 train_QA_roberta.py \
  --model_name_or_path ./QA_save_pretrained \
  --output_dir ./ckpt_QA \
  --do_predict \
  --context_file $1 \
  --test_file ./selection_pred.json \
  --output_file $3 \
  --gradient_accumulation_steps 2 \
  --per_gpu_train_batch_size 1 \
  --learning_rate 3e-5 \