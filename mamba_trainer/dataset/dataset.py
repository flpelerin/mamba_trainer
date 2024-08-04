from datasets import load_dataset

import torch
import numpy as np


from mamba_trainer.utils.metaclass import CallableMeta, Globals
from mamba_trainer.utils.util      import Util



class GenerateData(metaclass=CallableMeta):
    @staticmethod
    def __call__(dataset_path, tokenizer, seq_length, batch_size):
        dataset_path = dataset_path
        tokenizer    = tokenizer
        seq_length   = seq_length
        batch_size   = batch_size
        vocab_size   = len(tokenizer.vocab)

        dataset = load_dataset(dataset_path)

        texts = dataset["train"]["text"]
        text  = GenerateData.ConcatSplits(texts)

        input_ids = tokenizer.encode(text)
        input_ids = GenerateData.ClipOutOfVocab(input_ids, vocab_size)

        batches, num_batches =  GenerateData.BatchSequences(input_ids, seq_length, batch_size)
        GenerateData.Log(seq_length, num_batches, batch_size, batches)

        return batches, num_batches


    @staticmethod
    def Log(seq_length, num_batches, batch_size, batches):
        Util.Tee("config_log.txt", f"Dataset contains {num_batches} batches of {batch_size} sequences of {seq_length} tokens each ({Util.RoundNumber(seq_length * batch_size * num_batches)} tokens total)")
        Util.Tee("config_log.txt", f"Model's context window is {seq_length * batch_size} (seq_length * batch_size)")
        Util.Tee("config_log.txt", f"Batches shape is {torch.stack(batches).shape}")


    @staticmethod
    def ConcatSplits(texts):
        splits = [elem for sublist in texts for elem in sublist]
        text = ''.join(splits)
        return text


    @staticmethod
    def ClipOutOfVocab(input_ids, vocab_size):
        clipped = [min(token, vocab_size) for token in input_ids]
        return clipped


    @staticmethod
    def BatchSequences(input_ids, seq_length, batch_size):
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)

        num_batches = len(input_ids) // (seq_length * batch_size)
        total_elements = num_batches * seq_length * batch_size

        trimmed_array = input_ids[:total_elements]
        array_reshaped = trimmed_array.reshape((num_batches, batch_size, seq_length))

        tensor_batches = []
        for batch in array_reshaped:
            tensor_batch = torch.tensor(batch, dtype=torch.long).to(Util.GetDevice())
            tensor_batches.append(tensor_batch)

        return tensor_batches, num_batches
