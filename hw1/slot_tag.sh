# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --test_file "${1}" --ckpt_dir ckpt/slot/saved_weights.pt --pred_file "${2}"