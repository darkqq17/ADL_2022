python download_pretrain_model_mc.py \
    --model_name_or_path bert-base-chinese \
    --cache_dir ./MC/cache/ \

python download_pretrain_model_qa.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --cache_dir ./QA/cache/ \

wget https://www.dropbox.com/s/dbg4iyti8x2qcwh/MC_save_pretrained.zip?dl=1
wget https://www.dropbox.com/s/l29lsblh7v23pvo/ckpt_MC.zip?dl=1
wget https://www.dropbox.com/s/p9q1syilfac74il/QA_save_pretrained.zip?dl=1
wget https://www.dropbox.com/s/t2mu0e6lvtcjo99/ckpt_QA.zip?dl=1

unzip ./MC_save_pretrained.zip?dl=1
unzip ./ckpt_MC.zip?dl=1
unzip ./QA_save_pretrained.zip?dl=1
unzip ./ckpt_QA.zip?dl=1

