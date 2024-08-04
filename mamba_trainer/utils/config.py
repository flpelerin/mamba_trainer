
from dataclasses import dataclass, field

from mamba_trainer.utils.util import Util




@dataclass
class WandbConfig:
    entity=None,
    project=None,
    name='run-' + Util.RandomCode()
    api_key=None



@dataclass
class TrainEvent:
    enabled:  bool = False
    step:     int  = 0
    tee_file: str  = ''


@dataclass
class InferConfig:
    event:     TrainEvent = None
    model                 = None
    tokenizer               = None
    n_predict: int        = 0
    seed_text: str        = ''


@dataclass 
class Trainerconfig:
    log_config:   TrainEvent = None 
    infer_config: TrainEvent = None
    save_config:  TrainEvent = None
    wandb_config: WandbConfig = None




