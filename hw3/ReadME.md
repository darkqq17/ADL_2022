Reproduce my result
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl

Evaluation
python3 eval.py \
  -r <reference> \ 
  -s <submission> \
optional arguments:
  -h, --help 
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
  
Training
python3 summarization.py \
  --do_train \
  --do_eval \
  --model_name_or_path google/mt5-small \
  --output_dir ./output_dir \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --per_device_eval_batch_size=4 \
  --eval_accumulation_steps=4 \
  --predict_with_generate \
  --adafactor \
  --learning_rate 1e-3 \
  --warmup_ratio 0.1 \
  --num_train_epochs 5 \

Predicting
python3 summarization.py \
  --do_predict
  --model_name_or_path <model_name_or_path> \ 
  --test_file <test_file> \ 
  --output_file <output_file> \
  --output_dir <output_dir> \
  --predict_with_generate \
  --per_device_eval_batch_size 4 \
  --num_beams 5 \
  [--do_sample] \
  [--top_k <top_k>] \
  [--top_p <top_p>] \
  [--temperature <temperature>] \

Learning Curves
(human label approach)
Output 7 submission files for each checkpoints then run 7 times eval.py for each submission files to get the rouge score, and then write the score data into plot.ipynb to present the learning curves.
