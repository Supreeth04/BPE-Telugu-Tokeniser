import os
from collections import defaultdict
import csv
import json

def read_text(file, max_lines=1000):
    """Read text from CSV file with a limit on number of lines"""
    text = ''
    count = 0
    print(f"Reading file: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            text += ' '.join(row) + '\n'
            count += 1
            if count % 100 == 0:
                print(f"Read {count} lines...")
            if count >= max_lines:
                print(f"Reached max lines limit ({max_lines})")
                break
    print(f"Total lines read: {count}")
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
    print("Starting BPE training...")
    tokens = list(text.encode('utf-8'))
    ids = list(tokens)
    initial_vocab_size = len(set(tokens))
    print(f"Initial vocabulary size: {initial_vocab_size}")
    
    merges = {}
    next_id = 256  # Start after basic byte values
    
    iteration = 0
    while len(merges) + initial_vocab_size < vocab_size:
        iteration += 1
        if iteration % 100 == 0:
            print(f"Training iteration {iteration}, current vocab size: {len(merges) + initial_vocab_size}")
        
        pair_freqs = stats(ids)
        if not pair_freqs:
            print("No more pairs to merge")
            break
            
        pair = max(pair_freqs.items(), key=lambda x: x[1])
        if pair[1] < min_frequency:
            print(f"Stopping: pair frequency {pair[1]} below minimum {min_frequency}")
            break
            
        merges[pair[0]] = next_id
        next_id += 1
        
        ids = merge(ids, merges[pair[0]], pair[0])
        
        compression = len(tokens) / len(ids)
        if compression >= 3.2:
            print(f"Stopping: reached target compression ratio {compression:.2f}")
            break
    
    print(f"Training completed. Final vocab size: {len(merges) + initial_vocab_size}")
    return merges, compression

def save_merges(merges, filepath):
    """Save merges dictionary to a JSON file"""
    # Convert integer keys to strings since JSON only allows string keys
    serializable_merges = {str(pair): idx for pair, idx in merges.items()}
    print(serializable_merges)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_merges, f)

def load_merges(filepath):
    """Load merges dictionary from a JSON file"""
    print("loading merges")
    with open(filepath, 'r', encoding='utf-8') as f:
        serializable_merges = json.load(f)
    # Convert back to tuple keys and integer values
    merges = {(int(pair.split(',')[0][1:]), int(pair.split(',')[1][:-1])): idx 
             for pair, idx in serializable_merges.items()}
    print(merges)
    return merges

if __name__ == '__main__':
    # Read and train with smaller dataset
    print("Starting tokenizer training...")
    text = read_text('archive\\telugu_books\\telugu_books.csv')  # Limit to 1000 lines
    print(f"Text length: {len(text)} characters")
    
    merges, compression = train_bpe(text, vocab_size=2000)  # Reduced vocab size
    
    print("Saving trained merges...")
    save_merges(merges, 'trained_merges.json')
    
    print(f"\nFinal Statistics:")
    print(f"Vocabulary size: {len(merges) + 256}")
    print(f"Compression ratio: {compression:.2f}X")
    
    # Test with a smaller sample
    test_text = text[:100]  # Test with just 100 characters
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
