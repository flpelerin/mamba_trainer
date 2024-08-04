import os
import wandb

from mamba_trainer.utils.metaclass import CallableMeta, Globals
from mamba_trainer.utils.util      import Util




class Wandb(metaclass=CallableMeta):
    has_init = False

    @staticmethod
    def __call__(wandb_config=None):
        if wandb_config is None:
            return

        entity =  wandb_config.entity
        project = wandb_config.project
        name =    wandb_config.name
        api_key = wandb_config.api_key

        if entity is None or project is None or api_key is None:
            return

        os.environ['WANDB_API_KEY'] = api_key

        wandb.init(project, entity, name)
        Wandb.has_init = True


    @staticmethod
    def Log(args):
        if not Wandb.has_init:
            return

        wandb.log(args)


    @staticmethod
    def Finish():
        if not Wandb.has_init:
            return

        wandb.finish()
