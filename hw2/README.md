Multiple Choice
Training :
python3 train_MC.py \
    --model_name_or_path <model_name_or_path> \
    --output_dir ./MC_save_pretrained \
    --do_train \
    --do_eval \
    --train_file <train_file> \
    --validation_file <validation_file> \
    --context_file <context_file> \
    --cache_dir ./MC/cache/ \
    --gradient_accumulation_steps=2 \
    --per_gpu_train_batch_size=1 \
    --num_train_epochs=1 \
    --max_seq_length=512 \
    --learning_rate=3e-5 \
    --overwrite_output_dir \
    --save_pretrained ./MC_save_pretrained
Testing :
python3 train_MC.py \
    --output_dir ./MC_save_pretrained \
    --do_predict \
    --test_file <test_file> \
    --context_file <context_file> \
    --cache_dir ./MC/cache/ \
    --gradient_accumulation_steps=2 \
    --per_gpu_train_batch_size=1 \
    --num_train_epochs=1 \
    --max_seq_length=512 \
    --learning_rate=3e-5 \
    --model_name_or_path ./MC_save_pretrained \

Question Answering
Training :
python3 train_QA_roberta.py \
    --model_name_or_path <model_name_or_path> \
    --output_dir ./QA_save_pretrained \
    --do_train \
    --do_eval \
    --train_file <train_file> \
    --validation_file <validation_file> \
    --context_file <context_file> \
    --cache_dir ./QA/cache/ \
    --gradient_accumulation_steps=2 \
    --per_gpu_train_batch_size=1 \
    --num_train_epochs=3 \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --overwrite_output_dir \
    --save_pretrained ./QA_save_pretrained \

Testing :
python3 train_QA_roberta.py \
    --output_dir ./QA_save_pretrained \
    --do_predict \
    --test_file <test_file> \
    --context_file <context_file> \
    --gradient_accumulation_steps=2 \
    --per_gpu_train_batch_size=1 \
    --num_train_epochs=3 \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --model_name_or_path ./QA_save_pretrained \

Reproduce my result :
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.json