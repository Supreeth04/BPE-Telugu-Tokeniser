import os
from collections import defaultdict
import csv

def read_text(file):
    text = ''
    with open(file, 'r', encoding='utf-8') as f:  # Open the CSV file in text mode
        reader = csv.reader(f)
        for row in reader:
            text += ' '.join(row) + '\n'  # Join each row into a single string
    return text

def stats(ids):
    counts = defaultdict(int)
    if len(ids) < 2:
        return counts
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] += 1
    return counts

def merge(ids, index, pair):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            newids.append(index)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def encode(text, merges):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        pairs = stats(tokens)
        if not pairs:
            break
        pair = min(pairs.keys(), key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        tokens = merge(tokens, merges[pair], pair)
    return tokens

def decode(ids, merges):
    reverse_merges = {idx: pair for pair, idx in merges.items()}
    
    def expand_token(id):
        if id < 256:
            return [id]
        if id in reverse_merges:
            pair = reverse_merges[id]
            return expand_token(pair[0]) + expand_token(pair[1])
        print(f"Warning: Token ID {id} not found in reverse merges")
        return []
    
    bytes_list = []
    for id in ids:
        bytes_list.extend(expand_token(id))
    
    try:
        return bytes(bytes_list).decode('utf-8')
    except UnicodeDecodeError:
        return "[DECODE ERROR]"

def train_bpe(text, vocab_size=4000, min_frequency=2):
    tokens = list(text.encode('utf-8'))
    ids = list(tokens)
    initial_vocab_size = len(set(tokens))
    
    merges = {}
    next_id = 256  # Start after basic byte values
    
    while len(merges) + initial_vocab_size < vocab_size:
        pair_freqs = stats(ids)
        if not pair_freqs:
            break
            
        pair = max(pair_freqs.items(), key=lambda x: x[1])
        if pair[1] < min_frequency:
            break
            
        merges[pair[0]] = next_id
        next_id += 1
        
        ids = merge(ids, merges[pair[0]], pair[0])
        
        compression = len(tokens) / len(ids)
        if compression >= 3.2:
            break
    
    return merges, compression

if __name__ == '__main__':
    # Read and train
    text = read_text('archive\\telugu_books\\telugu_books.csv')  # Change to your CSV file name
    merges, compression = train_bpe(text)
    
    # Print statistics
    print(f"Vocabulary size: {len(merges) + 256}")  # 256 for byte tokens
    print(f"Compression ratio: {compression:.2f}X")
    
    # Test encoding and decoding
    test_text = text[:200]
    print("\nTesting with sample text:")
    print("Original:", test_text)
    
    encoded = encode(test_text, merges)
    print("\nEncoded (first 20 tokens):", encoded[:20])
    
    decoded = decode(encoded, merges)
    print("\nDecoded:", decoded)
    print("Successful roundtrip:", test_text == decoded)
    
    # Save vocabulary size and compression ratio
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(f"# Telugu BPE Tokenizer\n\n")
        f.write(f"Vocabulary size: {len(merges) + 256} tokens\n")
        f.write(f"Compression ratio: {compression:.2f}X\n")
