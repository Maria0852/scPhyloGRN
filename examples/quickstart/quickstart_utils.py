"""CPU scoring from a verified, fixed-context encoder cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inference_utils import StrongDecoder, DecoderNoBilinear, DecoderLinear


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class CachedPredictor:
    """Loads CPU embeddings and decoder only; never constructs a graph/encoder."""

    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name in ('metadata.json', 'embeddings.npy', 'decoder.pt'):
            if sha256(directory / name) != manifest['sha256'][name]:
                raise ValueError(f'Checksum mismatch: {name}. Restore the original bundle.')
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        self.genes = self.metadata['gene_names']
        if len(set(self.genes)) != len(self.genes):
            raise ValueError('Duplicate genes in cache metadata.')
        self.index = {gene: i for i, gene in enumerate(self.genes)}
        self.hidden = torch.from_numpy(np.load(directory / 'embeddings.npy', allow_pickle=False))
        config = self.metadata['model_config']
        if self.hidden.shape != (len(self.genes), config['out_dim']) or not torch.isfinite(self.hidden).all():
            raise ValueError('Invalid embedding dimensions or nonfinite embeddings.')
        size = config.get('decoder_hidden_dim', max(128, 2 * config['proj_dim']))
        kind = config['decoder']
        if kind == 'strong':
            self.decoder = StrongDecoder(config['out_dim'], size, config['dropout'])
        elif kind == 'no_bilinear':
            self.decoder = DecoderNoBilinear(config['out_dim'], size, config['dropout'])
        elif kind == 'linear':
            self.decoder = DecoderLinear(config['out_dim'])
        else:
            raise ValueError(f'Unsupported decoder: {kind}')
        state = torch.load(directory / 'decoder.pt', map_location='cpu', weights_only=True)
        self.decoder.load_state_dict(state, strict=True)
        self.decoder.eval()
        self.temperature = float(self.metadata['temperature'])
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Invalid calibration temperature.')

    def validate_pairs(self, pairs):
        columns = ['gene_i', 'gene_j']
        if not set(columns).issubset(pairs.columns) or pairs.empty:
            raise ValueError('Supply a nonempty table with gene_i and gene_j columns.')
        pairs = pairs[columns].copy()
        if pairs.isna().any().any():
            raise ValueError('Gene identifiers cannot be missing.')
        pairs = pairs.astype(str)
        if any((pairs[c].str.strip() != pairs[c]).any() or (pairs[c] == '').any() for c in columns):
            raise ValueError('Gene identifiers cannot be blank or contain surrounding whitespace.')
        unknown = sorted(set(pairs.to_numpy().ravel()) - set(self.index))
        if unknown:
            raise ValueError(f'Unknown genes: {unknown[:10]}. Match the exact cached gene identifiers.')
        if pairs.duplicated().any():
            raise ValueError('Duplicate directed pairs found. Remove them explicitly with drop_duplicates().')
        if (pairs.gene_i == pairs.gene_j).any():
            raise ValueError('Self-pairs were excluded from training candidates; remove them.')
        return pairs

    def predict(self, pairs, batch_size=4096):
        pairs = self.validate_pairs(pairs)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        edges = torch.tensor([[self.index[a], self.index[b]] for a, b in pairs.to_numpy()], dtype=torch.long)
        with torch.inference_mode():
            logits = torch.cat([self.decoder(self.hidden, edges[i:i + batch_size])
                                for i in range(0, len(edges), batch_size)])
            pairs['logit'] = logits.numpy()
            pairs['probability'] = torch.sigmoid(logits).numpy()
            pairs['calibrated_probability'] = torch.sigmoid(logits / self.temperature).numpy()
        return pairs
