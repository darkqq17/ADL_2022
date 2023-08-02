import json
from logging import exception
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, pad_to_len

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
max_length = 20

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    #print(intent2idx)
    tag2intent: Dict[int,str] = {v: k for k, v in intent2idx.items()}
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=datasets[TRAIN].collate_fn,
           pin_memory=True, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)
    #print(next(iter(train_dataloader)))
    
    valid_dataloader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=datasets[DEV].collate_fn,
           pin_memory=True, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        num_class=len(intent2idx)
    ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for data in train_dataloader:
            label_onehot = []
            for tag in data['tags']:
                label_onehot_cache = []
                for token in tag:
                    label_onehot_cache.append(intent2idx[token])
                label_onehot.append(label_onehot_cache)
            
            
            #print(label_onehot)
            optimizer.zero_grad()
            
            pad_length = 0
            for length in data['tags']:
                if len(length) > pad_length:
                    pad_length = len(length)
            
            pred_x = torch.tensor(vocab.encode_batch(data['tokens'], pad_length)).to(args.device)
            label_onehot = torch.tensor(pad_to_len(label_onehot,pad_length,intent2idx.get('O'))).to(args.device)
            
            #print(label_onehot)
            
            pred = model(pred_x)
            #print(pred)
            #print(pred.shape)
            #pred = layer(pred)
            loss_function = criterion(pred, label_onehot)
            
            loss_function.backward()
            optimizer.step()
        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_loss = 0
        pred_correct = 0
        dataset_size = len(valid_dataloader.dataset)
        best_valid_loss = float('inf')
        model.eval()       
        with torch.no_grad():
            for _, data in enumerate(valid_dataloader):
                label_onehot_total = []
                pred_total = []
                label_onehot = []
                for tag in data['tags']:
                    label_onehot_cache = []
                    label_onehot_total.append(tag) #label_onehot for seqeval
                    for token in tag:
                        label_onehot_cache.append(intent2idx[token])
                    label_onehot.append(label_onehot_cache)
                
                
                pad_length = 0
                for length in data['tags']:
                    if len(length) > pad_length:
                        pad_length = len(length)

                pred_x = torch.tensor(vocab.encode_batch(data['tokens'], pad_length)).to(args.device)
                label_onehot = torch.tensor(pad_to_len(label_onehot,pad_length,intent2idx.get('O'))).to(args.device)
                
                #print(data['intent'])
                pred = model(pred_x)
                #print(pred.argmax(1))
                #print(f"label_onehot:{label_onehot}")
                
                valid_loss += criterion(pred, label_onehot).item()
                
                pred = pred.permute(0,2,1)
                
                #pred_total for seqeval
                for id, text in enumerate(pred):
                    tokens_cache = []
                    for tokens in text[:len(data['tokens'][id])]:
                        tokens_cache.append(tag2intent.get(int(torch.argmax(tokens))))
                    pred_total.append(tokens_cache)
                    #print(pred_total)
                
                #print(pred.argmax(1))
                for x,y in zip(pred, label_onehot):
                    num_correct = (x.argmax(1) == y).type(torch.float).sum().item()
                    if num_correct == pad_length:
                        pred_correct += 1

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), str(args.ckpt_dir)+'/saved_weights.pt')
                
        pred_correct /= dataset_size
        tqdm.write(f"Joint Accuracy: {(100*pred_correct):>0.1f}%")
        
    #print(label_onehot_total)
    #print(pred_total)
    print(classification_report(label_onehot_total, pred_total, mode='strict', scheme=IOB2)) 
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)