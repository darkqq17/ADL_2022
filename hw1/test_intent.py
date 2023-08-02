import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from numpy import argmax

import torch
from torch.utils.data import DataLoader
import csv

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

max_lenth = 15

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2intent: Dict[int,str] = {v: k for k, v in intent2idx.items()}
    
    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=dataset.collate_fn,
           pin_memory=True, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    
    #print(next(iter(test_dataloader)))
    # TODO: predict dataset
    with torch.no_grad():
        x = []
        y = []
        for data in test_dataloader:
            pred_x = torch.tensor(vocab.encode_batch(data['text'], max_lenth)).to(args.device)
            intent_id = data['id']
            
            pred = model(pred_x)
            for id, text in enumerate(pred):
                x.append(intent_id[id])
                y.append(idx2intent.get(int(torch.argmax(text))))
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['id','intent'])
        for id, intent in zip(x, y):
            writer.writerow([id,intent])
            


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
