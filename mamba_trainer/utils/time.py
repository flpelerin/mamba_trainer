import time

from mamba_trainer.utils.metaclass import CallableMeta
from mamba_trainer.utils.metaclass import Globals



class Time(metaclass=CallableMeta):
    time_init = None
    time_last = None


    @staticmethod
    def __call__():
        return Time.Get()


    @staticmethod
    def Get():
        time_current = time.time()

        if Time.time_init == None:
            Time.time_init = time_current

        if Time.time_last == None:
            Time.time_last = time_current

        return time_current


    @staticmethod
    def FormatString(number):
        number = round(number)

        hours = number // 3600
        minutes = (number % 3600) // 60
        seconds = number % 60

        time_string = ""

        if hours > 0:
            time_string += str(hours) + "h "
        if minutes > 0 or hours > 0:
            time_string += str(minutes) + "m "
        time_string += str(seconds) + "s"

        return time_string


    @staticmethod
    def FormatSecond(time_string):
        number = 0
        parts = time_string.split()

        for part in parts:
            if part.endswith('h'):
                number += int(part[:-1]) * 3600
            elif part.endswith('m'):
                number += int(part[:-1]) * 60
            elif part.endswith('s'):
                number += int(part[:-1])

        return number


    @staticmethod
    def Up(raw=False):
        time_current = Time()
        time_up = time_current - Time.time_init

        return time_up if raw is True else Time.FormatString(time_up)


    @staticmethod
    def Step(raw=False):

        time_current = Time()
        time_step = time_current - Time.time_last
        Time.time_last = time_current

        return time_step if raw is True else Time.FormatString(time_step)
