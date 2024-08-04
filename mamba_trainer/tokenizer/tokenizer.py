from collections import Counter

import struct



def most_occurring_pair(arr):
    # Create pairs of adjacent elements, ignoring pairs where the second element > 255
    pairs = [(a, b) for a, b in zip(arr[:-1], arr[1:]) if b <= 255]

    pair_counts = Counter(pairs)
    if not pair_counts:
        return None

    most_common_pair = max(pair_counts, key=pair_counts.get)
    return list(most_common_pair)



def replace_pair(initial_list, pair_to_remove, replace_with):
    result = []
    i = 0

    while i < len(initial_list):
        if i + 1 < len(initial_list) and initial_list[i:i+2] == pair_to_remove:
            result.append(replace_with)
            i += 2

        else:
            result.append(initial_list[i])
            i += 1

    return result



class Token:
    def __init__(self, byte, prev):
        self.byte = byte
        self.prev = prev

    def pack(self):
        return struct.pack("=B H", ord(self.byte), self.prev)

    def __str__(self):
        return f"{self.byte}, {self.prev}"

    def to_binary(self):
        return self.pack()

    @classmethod
    def from_binary(cls, data):
        if len(data) != 3:
            raise ValueError("Data has invalid length, Exprected 3 bytes.")

        byte, prev = struct.unpack("=B H", data)
        return cls(chr(byte), prev)


class Vocab:
    def __init__(self):
        self.clear()

    def __getitem__(self, id):
        return self.tokens[id]

    def __setitem__(self, id, token):
        self.tokens[id] = token

    def clear(self):
        self.tokens = []
        self.vocab_size = 0

    def __len__(self):
        return self._get_size()

    def _get_size(self):
        return self.vocab_size

    def __iadd__(self, token):
        self._add_token(token)
        return self

    def __add__(self, token):
        return self._add_token(token)

    def _add_token(self, token):
        self.tokens.append(token)
        self.vocab_size += 1
        return self.vocab_size - 1

    def find(self, byte, prev):
        for i in range(self.vocab_size):
            token = self.tokens[i]

            if byte == token.byte and prev == token.prev:
                return i

        return 0

    def __str__(self):
        text = '['
        n_tokens = len(self.tokens)

        for i in range(n_tokens):
            text += '{' + str(self.tokens[i]) + ('}, ' if i < n_tokens - 1 else '}]')

        return text


    def to_file(self, file):
        with open(file, 'ab') as f:
            for token in self.tokens:
                f.write(token.to_binary())

    def from_file(self, file):
        self.clear()

        with open(file, 'rb') as f:
            while True:
                try:
                    data = f.read(3)
                    token = Token.from_binary(data)
                    self += token

                except ValueError:
                    break


class Tokenizer:
    def __init__(self):
        self.vocab = Vocab()
        self._init_byte_level()

    def _init_byte_level(self):
        self.vocab.clear()

        for i in range(256):
            token = Token(chr(i), 0)
            self.vocab += token


    def train(self, text, target_length=None):
        arr = [ord(c) for c in text]

        while True:
            if target_length is not None:
                if len(self.vocab) >= target_length:
                    break

            pair = most_occurring_pair(arr)

            if arr is None or pair is None:
                break;

            byte = chr(pair[1])
            prev = pair[0]
            token = Token(byte, prev)
            id = self.vocab + token

            arr = replace_pair(arr, pair, id)


    def _decode_one(self, id):
        text = ""
        while True:
            token = self.vocab[id]

            text += token.byte
            if token.prev == 0:
                break

            id = token.prev

        return text[::-1]

    def decode(self, ids):
        if not isinstance(ids, list):
            ids = [ids]

        text = ""
        for id in ids:
            text += self._decode_one(id)

        return text

    def _encode_one(self, text):
        prev = 0

        for i in range(len(text)):
            next = self.vocab.find(text[i], prev)
            if next == 0:
                return prev, text[i:]

            prev = next

        return prev, ""

    def encode(self, text):
        if isinstance(text, list):
            texts = text
            text = ""

            for i in range(len(texts)):
                text += texts[i]

        ids = []

        while text != "":
            id, text = self._encode_one(text)

            if id == 0:
                text = text[1:]

            ids.append(id)

        return ids


    def add_one_special_token(self, text):
        prev = 0
        byte = None

        for i in range(len(text)):
            byte = text[i]
            token = self.vocab.find(byte, prev)

            if token:
                prev = token
                continue

            token = Token(byte, prev)
            prev = self.vocab + token

        return prev

    def add_special_token(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        for i in range(len(texts)):
            text = texts[i]
            self.add_one_special_token(text)

    def __str__(self):
        return str(self.vocab)

    def to_file(self, file):
        self.vocab.to_file(file)

    def from_file(self, file):
        self.vocab.from_file(file)
