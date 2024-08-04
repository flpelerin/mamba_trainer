import torch

from mamba_trainer.utils.metaclass import CallableMeta, Globals
from mamba_trainer.utils.wandb     import Wandb
from mamba_trainer.utils.time       import Time
from mamba_trainer.utils.util      import Util




from dataclasses import dataclass, field

from mamba_trainer.utils.metaclass import CallableMeta, Globals
from mamba_trainer.utils.util      import Util



@dataclass
class WandbConfig:
    entity=None,
    project=None,
    name='run-' + Util.RandomCode()
    api_key=None





@dataclass
class Event:
	enabled:  bool = False
	step:     int  = 0
	tee_file: str  = ''


@dataclass 
class EventConfig:
	log_config:   Event = None 
	infer_config: Event = None
	save_config:  Event = None




default_config = EventConfig(
	log_config = Event(
		enabled = True,
		step = 10,
		tee_file = 'training_log.txt'
	),
	infer_config = Event(
		enabled = True,
		step = 100,
		tee_file = 'inference_log.txt'
	),
	save_config = None
)




class TrainModel(metaclass=CallableMeta):
    train_step:   int         = 0
    event_config: EventConfig = None


    @staticmethod
    def __call__(model, batches, num_batches, num_epochs=10, learning_rate=1e-4, wandb_config=None, event_config = default_config):
        TrainModel.event_config = event_config

        Wandb(wandb_config)
        TrainModel.Train(model, batches, num_batches, num_epochs, learning_rate)
        Wandb.Finish()
		
		model.save()


    @staticmethod
    def Train(model, batches, num_batches, num_epochs, learning_rate):
        optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion     = torch.nn.CrossEntropyLoss()

        model.train()

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                input_ids = batches[batch]

                loss = model.compute_loss(input_ids)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                TrainModel.LogStep(epoch, num_epochs, batch, num_batches, loss)

    @staticmethod
    def ComputeTime(step, num_epochs, num_batches):
        time_step       = Time.Step(raw=True)
        time_up         = Time.Up(raw=True)
        time_per_epoch  = time_step * num_batches * num_epochs
        time_remain     = time_per_epoch - time_up if time_up < time_per_epoch else 0

        return Time.FormatString(time_up), Time.FormatString(time_per_epoch), Time.FormatString(time_remain)

    @staticmethod
    def LogStep(epoch, num_epochs, batch, num_batches, loss, log_every=10):
        step = TrainModel.train_step
        TrainModel.train_step += 1

        loss = loss.item()

        wandb_args = {"step": step, "epoch": epoch, "batch": batch, "loss": loss}
        Wandb.Log(wandb_args)

		event_config = TrainModel.event_config
		if event_config is None:
			return

		with event = event_config.log_config:
			if event.enabled is True:
				if event.step % step == 0:
					time_up, time_per_epoch, time_remain = TrainModel.ComputeTime(step, num_epochs, num_batches)
	            	Util.Tee(event.tee_file, f"Step: {step}\t\tEpoch: {epoch} / {num_epochs}\t\tBatch: {batch} / {num_batches}\t\tLoss: {round(loss, 4)}\t\tTime: {time_up} / {time_per_epoch}\t({time_remain} remaining)")

		with event = event_config.infer_config:
			if event.enabled is True:
				if event.step % step == 0:
					Util.Tee(event.tee_file, f"{model.generate_text(Globals.tokenizer, Globals.seed_text, Globals.num_predict)}\n")


		# Implement save to tee_file here
