import logging
import os
import sys
import json
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
#from tw_rouge import get_rouge

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
       default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default="google/mt5-small", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="google/mt5-small", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    train_file: Optional[str] = field(
        default="./data/train.jsonl", metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default="./data/public.jsonl",
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional output file to predict test file" "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to use sampling ; use greedy decoding otherwise. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={
            "help": "The value used to module the next token probabilities. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    #download dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
        if extension == 'jsonl':
            extension = 'json'
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
        if extension == 'jsonl':
            extension = 'json'
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
        if extension == 'jsonl':
            extension = 'json'
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir
    )
    
    # print(raw_datasets['train']['title'][0])
    # print(raw_datasets['train']['maintext'][0])
    
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    #print(model)
    
    #Preprocssing the datasets
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    
    #print(column_names)
    
    text_colunm = column_names[3]
    summary_column = column_names[1]
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    def preprocess_function(examples):
        #remove pairs where at least one record is None
        
        inputs, target = [], []
        for i in range(len(examples[text_colunm])):
            if examples[text_colunm][i] and examples[summary_column][i]:
                inputs.append(examples[text_colunm][i])
                target.append(examples[summary_column][i])
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        #print(model_inputs)
        
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=target, max_length=max_target_length, padding=padding, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        #print(model_inputs)
        return model_inputs

    
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file= not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset"
        )
    
    if training_args.do_eval:
        eval_dataset = raw_datasets["test"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        
    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        predict_dataset_id = predict_dataset["id"]
        #print(predict_dataset_id)
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
    
    #Data collator
    #label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    #metric, huggingface rouge
    metric = evaluate.load("rouge")
    

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    #print(train_dataset[0])
    
    #Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #Trainning
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        if not data_args.do_sample:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict", 
                max_length=max_length, 
                num_beams=num_beams
            )
            
            # metrics = predict_results.metrics
            # max_predict_samples = (
            #     data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            # )
            # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            # trainer.log_metrics("predict", metrics)
            # trainer.save_metrics("predict", metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    output_prediction_file =  data_args.output_file
                    with open(output_prediction_file, "w") as writer:
                        for predict, id in zip(predictions, predict_dataset_id):
                            predict_format = {"title": predict, "id": id}
                            writer.write(json.dumps(predict_format))
                            writer.write("\n")

    #Predict with Top-k/Top-p
    if training_args.do_predict:
        if data_args.do_sample:
            
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8
            )

            test_dataloader = DataLoader(
                predict_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
            )
            #print(next(iter(test_dataloader)))

            gen_kwargs = {
                "max_length": data_args.val_max_target_length,
                "do_sample" : data_args.do_sample,
                "top_k" : data_args.top_k,
                "top_p" : data_args.top_p,
                "temperature" : data_args.temperature,
            }
            
            prediction = []
            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    batch.to("cuda")
                    generated_tokens = model.generate(
                        batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                        **gen_kwargs
                    )
                    generated_tokens = generated_tokens.cpu().numpy()
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    for pred in decoded_preds:
                        prediction.append(pred.strip())
                    
            output_prediction_file =  data_args.output_file
            with open(output_prediction_file, "w") as writer:
                for predict, id in zip(prediction, predict_dataset_id):
                    predict_format = {"title": predict, "id": id}
                    writer.write(json.dumps(predict_format))
                    writer.write("\n")
    
if __name__ == "__main__":
    main()