from mamba_trainer.util.metaclass import CallableMeta, Globals
from mamba_trainer.util.wandb     import Wandb
from mamba_trainer.util.time 	  import Time
from mamba_trainer.util.util      import Util


class TrainModel(metaclass=CallableMeta):
    train_step = 0

    @staticmethod
    def __call__(model, batches, num_epochs, learning_rate):
        optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion     = torch.nn.CrossEntropyLoss()

        model.train()
        Wandb.Init()

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                input_ids = batches[batch]

                loss = model.compute_loss(input_ids)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                TrainModel.LogStep(epoch, num_epochs, batch, num_batches, loss)

        Wandb.Finish()

    @staticmethod
    def ComputeTime(step, num_epochs, num_batches):
        time_step       = Time.Step(raw=True)
        time_up         = Time.Up(raw=True)
        time_per_epoch  = time_step * num_batches * num_epochs
        time_remain     = time_per_epoch - time_up if time_up < time_per_epoch else 0

        return Time.FormatString(time_up), Time.FormatString(time_per_epoch), Time.FormatString(time_remain)

    @staticmethod
    def LogStep(epoch, num_epochs, batch, num_baches, loss, log_every=10):
        step = TrainModel.train_step
        loss = loss.item()

        time_up, time_per_epoch, time_remain = TrainModel.ComputeTime(step, num_epochs, num_batches)

        wandb_args = {"step": step, "epoch": epoch, "batch": batch, "loss": loss}
        Wandb.Log(wandb_args)

        if step % log_every == 0:
            Util.Tee("training_log.txt", f"Step: {step}\t\tEpoch: {epoch} / {num_epochs}\t\tBatch: {batch} / {num_batches}\t\tLoss: {round(loss, 4)}\t\tTime: {time_up} / {time_per_epoch}\t({time_remain} remaining)")

        if Globals.infer_during_training is True and step % (log_every * 10) == 0:
            Util.Tee("inference_log.txt", f"{model.generate_text(Globals.tokenizer, Globals.seed_text, Globals.num_predict)}\n")

        TrainModel.train_step += 1