import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from model import SeqClassifier

import torch
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
max_lenth = 20
print(torch.cuda.is_available())


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=datasets[TRAIN].collate_fn,
           pin_memory=True, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)
    
    valid_dataloader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=datasets[DEV].collate_fn,
           pin_memory=True, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    #print(embeddings)
    
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
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
            for intent in data['intent']:
                #print(intent2idx[intent])
                label_onehot.append(intent2idx[intent])
            
            #print(label_onehot)
            optimizer.zero_grad()
            
            layer = torch.nn.BatchNorm1d(150)
            
            pad_length = 0
            for length in data['intent']:
                if len(length) > pad_length:
                    pad_length = len(length)
            
            pred_x = torch.tensor(vocab.encode_batch(data['text'], pad_length)).to(args.device)
            label_onehot = torch.tensor(label_onehot).to(args.device)
            
            #print(pred)
            #print(label_onehot)
            
            pred = model(pred_x)
            #pred = layer(pred)
            loss_function = criterion(pred, label_onehot)
            
            loss_function.backward()
            optimizer.step()
        # TODO: Evaluation loop - calculate accuracy and save model weights
        valid_loss = 0
        pred_correct = 0
        dataset_size = len(valid_dataloader.dataset)
        #print(len(valid_dataloader.dataset))
        best_valid_loss = float('inf')
        model.eval()       
        with torch.no_grad():
            for _, data in enumerate(valid_dataloader):
                label_onehot = []
                for intent in data['intent']:
                    label_onehot.append(intent2idx[intent])
                    
                pad_length = 0
                for length in data['intent']:
                    if len(length) > pad_length:
                        pad_length = len(length)

                pred_x = torch.tensor(vocab.encode_batch(data['text'], pad_length)).to(args.device)
                label_onehot = torch.tensor(label_onehot).to(args.device)
                
                #print(data['intent'])
                pred = model(pred_x)
                #print(pred.argmax(1))
                #print(f"label_onehot:{label_onehot}")
                valid_loss += criterion(pred, label_onehot).item()
                #print(pred.argmax(1))
                pred_correct += (pred.argmax(1) == label_onehot).type(torch.float).sum().item()
                
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), str(args.ckpt_dir)+'/saved_weights.pt')
                
                '''_, pred = torch.max(pred.data, 1)
                label_onehot = torch.argmax(label_onehot,dim=-1)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), str(args.ckpt_dir)+'/saved_weights.pt')
                
                label_onehot = label_onehot.cpu().numpy()
                pred = pred.cpu().numpy()
                
                f1_macro = f1_score(label_onehot, pred, average='macro')
                val_acc = accuracy_score(label_onehot, pred)'''
                
        pred_correct /= dataset_size
        tqdm.write(f"Accuracy: {(100*pred_correct):>0.1f}%")  
                     
        #tqdm.write(f"Accuracy: {100*val_acc:.2f}%\n")

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
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
    parser.add_argument("--num_epoch", type=int, default=500)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
