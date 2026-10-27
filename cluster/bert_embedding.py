"""Drop-in replacement for the `bert-embedding` package, for machines where it can't be installed.

`bert-embedding` runs on mxnet, which has no ARM64 (DGX Spark) builds and is no longer maintained.
This reimplements the one call LViT's Load_Dataset.py makes, with the same weights (Google's
BERT-base uncased, loaded through HuggingFace transformers) and the same output format:

    BertEmbedding()(sentences) -> [(tokens, [768-d np.ndarray per token]), ...]   # one entry per sentence

Matching bert-embedding's defaults: max_seq_length=25 (incl. [CLS]/[SEP]), [CLS]/[SEP] dropped,
last-layer outputs, and '##' word pieces averaged into the word they belong to (oov_way='avg').
Runs on CPU (as the original did).

Running BERT per sample inside DataLoader workers is very slow (and can hang in forked workers),
so prepare_run.sh patches Load_Dataset.py to call precompute() with every description once, in the
main process; after that each call is a dictionary lookup.

Copied next to train_model.py by prepare_run.sh, so `from bert_embedding import BertEmbedding` finds it.
"""
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

MODEL_NAME = "bert-base-uncased"


class BertEmbedding:
    def __init__(self, max_seq_length=25):
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertModel.from_pretrained(MODEL_NAME).eval()
        self.cache = {}

    def precompute(self, texts):
        """Embed every sentence of every text now, so later calls are lookups (texts are split on '\\n' like Load_Dataset does)."""
        sentences = sorted({s for t in texts for s in t.split('\n')} - set(self.cache))
        print('BERT: embedding %d unique sentences...' % len(sentences), flush=True)
        self(sentences)
        print('BERT: done', flush=True)

    @torch.no_grad()
    def __call__(self, sentences, oov_way="avg"):
        results = []
        for sentence in sentences:
            if (sentence, oov_way) in self.cache:
                results.append(self.cache[(sentence, oov_way)])
                continue
            enc = self.tokenizer(sentence, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
            hidden = self.model(**enc).last_hidden_state[0].numpy()
            pieces = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])[1:-1]  # drop [CLS], [SEP]
            hidden = hidden[1:-1]

            tokens, groups = [], []
            for piece, vec in zip(pieces, hidden):
                if piece.startswith("##") and tokens:
                    tokens[-1] += piece[2:]
                    groups[-1].append(vec)
                else:
                    tokens.append(piece)
                    groups.append([vec])
            if oov_way == "avg":
                embeddings = [np.mean(g, axis=0) for g in groups]
            elif oov_way == "sum":
                embeddings = [np.sum(g, axis=0) for g in groups]
            else:  # 'last'
                embeddings = [g[-1] for g in groups]
            self.cache[(sentence, oov_way)] = (tokens, embeddings)
            results.append((tokens, embeddings))
        return results
