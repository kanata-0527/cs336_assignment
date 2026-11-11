import json

import regex as re

from cs336_basics.train_bpe import PAT
from typing import Iterator, Iterable

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self.special_ids: dict[str, int] = {}

        next_id = max(self.vocab) + 1 if self.vocab else 0
        for token in special_tokens or []:
            if not token:
                raise ValueError("Special tokens must be non-empty.")
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_id] = token_bytes
                self.token_to_id[token_bytes] = next_id
                next_id += 1
            self.special_ids[token] = self.token_to_id[token_bytes]

        self._special_tokens = sorted(self.special_ids, key=len, reverse=True)
        self._special_lookahead = len(self._special_tokens[0]) - 1 if self._special_tokens else 0

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            for key, value in json.load(file).items():
                vocab[key] = bytes.fromhex(value)

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as file:
            for line in file:
                pair = tuple(bytes.fromhex(token) for token in line.strip().split())
                if len(pair) == 2:
                    merged_token = pair[0] + pair[1]
                    merges.append((pair, merged_token))

        return cls(vocab, merges, special_tokens)

    def _next_special(self, text: str, start: int) -> tuple[int, str | None]:
        special_start = len(text)
        next_special = None
        # Longer tokens are checked first, so a match at the same position keeps the first one.
        for token in self._special_tokens:
            position = text.find(token, start)
            if 0 <= position < special_start:
                special_start = position
                next_special = token
                if position == start:
                    break
        return special_start, next_special

    def encode(self, text: str) -> list[int]:
        ids = []
        start = 0

        while start < len(text):
            special_start, next_special = self._next_special(text, start)

            ordinary_text = text[start:special_start]
            for match in re.finditer(PAT, ordinary_text):
                tokens = [bytes([value]) for value in match.group().encode("utf-8")]
                while len(tokens) > 1:
                    best_pair = None
                    best_rank = None
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        rank = self.merge_rank.get(pair)
                        if rank is not None and (best_rank is None or rank < best_rank):
                            best_pair = pair
                            best_rank = rank

                    if best_pair is None:
                        break

                    new_tokens = []
                    merged_token = best_pair[0] + best_pair[1]
                    i = 0
                    while i < len(tokens):
                        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                            new_tokens.append(merged_token)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens

                ids.extend(self.token_to_id[token] for token in tokens)

            if next_special is None:
                break
            ids.append(self.special_ids[next_special])
            start = special_start + len(next_special)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for chunk in iterable:
            pending += chunk
            start = 0
            # 保存最后几个字符，以便在下一个 chunk 中继续匹配特殊 token
            limit = max(0, len(pending) - self._special_lookahead)

            while start < limit:
                position, special = self._next_special(pending, start)
                if special is None or position >= limit:
                    break

                for match in re.finditer(PAT, pending, pos=start, endpos=position):
                    yield from self.encode(match.group())
                yield self.special_ids[special]
                start = position + len(special)

            # Keep the last two matches in case the next chunk changes them.
            if start < limit:
                tail = []
                for match in re.finditer(PAT, pending, pos=start, endpos=limit):
                    tail.append(match)
                    if len(tail) > 2:
                        ready = tail.pop(0)
                        yield from self.encode(ready.group())
                        start = ready.end()

            pending = pending[start:]

        yield from self.encode(pending)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[token_id] for token_id in ids)
        return raw.decode("utf-8", errors="replace")
