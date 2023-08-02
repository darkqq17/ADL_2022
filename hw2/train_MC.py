import json
import logging
import os
from typing import Optional, Union
from itertools import chain
from dataclasses import dataclass, field

import datasets
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModelForMultipleChoice,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default="./ckpt_MC",
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="./ckpt_MC",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    save_pretrained: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    context_file : Optional[str] = field(
        default=None,
        metadata={"help": "An optional input context data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
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
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional output file."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    
@dataclass
class DataCollatorForMultipleChoice:
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    pass
    

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #data preprocessing
    with open(data_args.context_file,'r',encoding="utf-8") as f:
        df_context = json.load(f)
    
    if training_args.do_train:
        #download the datasets
        data_files = {"train":data_args.train_file, "validate":data_args.validation_file}
        extension = data_args.train_file.split(".")[-1]
        ###print(extension)
        raw_datasets = load_dataset(
            extension,
            data_files=data_files
        )
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validate"]
        ###print(raw_datasets)
        ###print(context_datasets["context"][0])
    
    if training_args.do_predict:
        extension = data_args.test_file.split(".")[-1]
        test_dataset = load_dataset(
            extension,
            data_files={ "test": data_args.test_file }
            )
    
    # Load pretrained model and tokenizer
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
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warn(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    #preprocess dataset
    def preprocess_function(examples):
        '''first_sentences = [[question] * 7 for question in examples["question"]]
        paragraphs_idx = [idx + random.sample(set(range(len(df_context))) - set(idx), 7 - len(idx)) for idx in examples["paragraphs"]]
        second_sentences = [
            [df_context[i] for i in idx] for idx in paragraphs_idx
        ]'''
        labels = []
        question_sequence = [[question]*4 for question in examples["question"]]
        answer_sequence = [[df_context[id] for id in paragraphs] for paragraphs in examples["paragraphs"]]
        ###print(len(examples["paragraphs"]))
        for number in range(len(examples["paragraphs"])):
            labels.append(examples["paragraphs"][number].index(examples["relevant"][number]))
        ###print(labels)
        ###print(paragraphs_idx)

        question_sequence = list(chain(*question_sequence))
        answer_sequence = list(chain(*answer_sequence))
        ###print(question_sequence)
        ###print(answer_sequence)
        
        tokenized_examples = tokenizer(
            question_sequence,
            answer_sequence,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        '''tokenized_inputs = {k: [v[i : i + 7] for i in range(0, len(v), 7)] for k, v in tokenized_examples.items()}
        if "relevant" in examples.keys():
            relevant = examples["relevant"]
            tokenized_inputs['label'] = [paragraphs.index(rel) for rel, paragraphs in zip(relevant, paragraphs_idx)]
        else:
            tokenized_inputs['label'] = [0 for _ in paragraphs_idx]'''
        return tokenized_inputs
    
    def preprocess_test_function(examples):
        question_sequence = [[question]*4 for question in examples["question"]]
        paragraphs_idx = examples["paragraphs"]
        answer_sequence = [[df_context[id] for id in paragraphs] for paragraphs in examples["paragraphs"]]
        
        question_sequence = list(chain(*question_sequence))
        answer_sequence = list(chain(*answer_sequence))
        
        tokenized_examples = tokenizer(
            question_sequence,
            answer_sequence,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True,
        )
        
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = [0 for _ in paragraphs_idx]
        return tokenized_inputs
    
    if training_args.do_train:
        '''train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")'''
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_eval:
        '''eval_dataset = datasets["validation"]
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")'''
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    ###print(train_dataset)
    ###print(eval_dataset)
    if training_args.do_predict:
        '''test_dataset = datasets["test"]
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")'''
        test_dataset = test_dataset.map(
            preprocess_test_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    ###print(test_dataset)
    
    
    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer)
    )
    
    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    '''processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
        )
    print(processed_datasets["train"]["input_ids"][0])
    print(processed_datasets["train"]["token_type_ids"][0])
    print(processed_datasets["train"]["attention_mask"][0])
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validate"]'''
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        ###print(last_checkpoint)
        
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
    ###print(training_args.resume_from_checkpoint)
    ###print(last_checkpoint)
    # Training
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
        
        model.save_pretrained(model_args.save_pretrained)
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    ###print(test_dataset['test'])
    # Predict
    if training_args.do_predict:
        logger.info("*** Predict***")
        
        results = trainer.predict(test_dataset['test'])
        preds = np.argmax(results.predictions, axis=1)
        output_json = []
        for i, pred in enumerate(tqdm(preds)):
            ex = {
                'id': test_dataset['test']['id'][i],
                'question': test_dataset['test']['question'][i],
                'paragraphs': test_dataset['test']['paragraphs'][i],
                'relevant': test_dataset['test']['paragraphs'][i][pred] if pred < len(test_dataset['test']['paragraphs'][i]) else test_dataset['test']['paragraphs'][0]
            }
            # if 'answers' in test_dataset.features:
            #     ex['answers'] = test_dataset['answers'][i]
            output_json.append(ex)
        json.dump(output_json, open(data_args.output_file, 'w',encoding='utf-8'), indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    main()