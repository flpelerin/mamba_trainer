from types import MethodType

import torch

from mamba_trainer.utils.metaclass import CallableMeta
from mamba_trainer.utils.util import Util 



class GenerateModel(metaclass=CallableMeta):
    @staticmethod
    def __call__(params, model_class, config_class):
        config = config_class(**params)
        model = model_class(config).to(Util.GetDevice())

        model.compute_loss = MethodType(GenerateModel.AutoRegressiveLossFunction, model)
        model.generate_text = MethodType(GenerateModel.GenerateText, model)
        model.save = MethodType(GenerateModel.SaveToPytorch, model)

        GenerateModel.Log(model)

        return model, config

    @staticmethod
    def Log(model):
        model_size, rounded_model_size = Util.GetNumParams(model)
        Util.Tee("config_log.txt", f"Model has {model_size} ({rounded_model_size}) parameters")

    @staticmethod
    def AutoRegressiveLossFunction(self, input_ids, labels=None, criterion=None):
        model = self
        lm_logits = model(input_ids).logits

        labels = input_ids.to("cuda")
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = criterion or torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    @staticmethod
    def GenerateText(self, tokenizer, seed_text, num_predict):
        model = self
        max_len = num_predict + len(seed_text)

        with torch.no_grad():
            encoded_ids = tokenizer.encode(seed_text)
            input_ids = torch.tensor(encoded_ids).unsqueeze(0).to(Util.GetDevice())
            output = model.generate(input_ids, max_length=max_len)

            logits = output[0].tolist()
            text = tokenizer.decode(logits)
        return text

    @staticmethod
    def SaveToPytorch(self):
        model = self
        model.save_pretrained('./')

