import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_pretokenized_file( # 处理输入文件，统计字节对频率
    input_path: str,
    special_tokens: list[str]
) -> Counter:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text]

    if special_tokens:
        separators = []
        for token in special_tokens:
            separators.append(re.escape(token))

        pattern = "|".join(separators)
        chunks = re.split(pattern, text) # 先用正则表达式将文本分割成多个块，然后对每个块进行预分词处理

    counts = Counter() # 表示一个字典，用于统计每个预分词的出现次数

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretoken = match.group()
            counts[pretoken] += 1

    return counts
    
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    for token in range(256):
        vocab[token] = bytes([token])

    # Add special tokens to the vocabulary
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    counts = read_pretokenized_file(input_path, special_tokens)

    word_counts = Counter()
    # 将 counts 中的每个预分词转换为字节表示，并统计其出现次数
    for pretoken, count in counts.items(): 
        tokens = []
        for byte_value in pretoken.encode("utf-8"):
            tokens.append(vocab[byte_value])

        word_counts[tuple(tokens)] += count

    merges = []

    # 对每个 pair 的出现次数进行统计，存储在 pair_counts 中
    pair_counts = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}    # 优化：索引并缓存所有 pair counts，每轮增量更新，提速
    for tokens, count in word_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(tokens)  # 缓存 pair 对应的 word 列表，便于后续增量更新

    if not pair_counts:
        return vocab, merges
    
    while len(vocab) < vocab_size and pair_counts:
        # 先比较 count,再比较 pair 的字典序
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]

        affected_words = tuple(pair_to_words[best_pair]) # 受影响的 word 列表，便于后续增量更新

        for tokens in affected_words:
            count = word_counts.pop(tokens)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if tokens[i:i + 2] == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            for i in range(len(tokens) - 1): # 受影响的 word 中的 pair 出现次数需要减去
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] -= count
                pair_to_words[pair].discard(tokens) 

                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    del pair_to_words[pair]

            new_tokens_tuple = tuple(new_tokens)
            word_counts[new_tokens_tuple] += count

            for i in range(len(new_tokens) - 1): # 受影响的 word 中的 pair 出现次数需要加上
                pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[pair] += count
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(new_tokens_tuple)

        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

    return vocab, merges