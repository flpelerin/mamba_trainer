import torch
import numpy as np


class Util:
    @staticmethod
    def GetDevice():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @staticmethod
    def RoundNumber(number):
        suffixes = ['', 'k', 'm', 'b']

        if number < 1000:
            return str(number)

        magnitude = 0
        while abs(number) >= 1000:
            magnitude += 1
            number /= 1000.0

        return '{:.0f}{}'.format(number, suffixes[magnitude])


    @staticmethod
    def RandomCode():
        import math
        import random

        code = '';
        chars = '0123456789abcdef'
        count = 8;

        for i in range(0, count):
          code += chars[math.floor(random.randrange(len(chars)))]

        return code


    @staticmethod
    def GetNumParams(model):
        size = sum(p.numel() for p in model.parameters())
        rounded_size = Util.RoundNumber(size)

        return size, rounded_size


    @staticmethod
    def Tee(file, str):
        print(str)

        with open(file, 'a') as f:
            f.write(f"{str}\n")
