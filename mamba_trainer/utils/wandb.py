import os
import wandb

from mamba_trainer.utils.metaclass import CallableMeta, Globals



class Wandb(metaclass=CallableMeta):
    wandb_has_init = False

    @staticmethod
    def Init():
        if Globals.wandb_log_run is False:
            return

        if Wandb.wandb_has_init is True:
            return

        Wandb.wandb_has_init = True

        project  = Globals.wandb_project
        entity   = Globals.wandb_entity
        api_key  = Globals.wandb_api_key
        name     = Globals.wandb_name

        if name is None or name == "":
            name = "run-" + Util.RandomCode()

        os.environ["WANDB_API_KEY"] = api_key

        wandb.init(project=project, entity=entity, name=name)


    @staticmethod
    def Log(args):
        if Globals.wandb_log_run is False:
            return

        if Wandb.wandb_has_init is False:
            Wandb.Init()

        if Wandb.wandb_has_init is True:
            wandb.log(args)


    @staticmethod
    def Finish():
        if Globals.wandb_log_run is False:
            return

        if Wandb.wandb_has_init is True:
            wandb.finish()
