import timm
from tabulate import tabulate
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils.tools import count_num_parameters


@MODEL_REGISTRY.register()
class ERM(Trainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def build_model(self):
        self.model = timm.create_model(
            self.cfg.MODEL.ERM.BACKBONE,
            pretrained=True,
            num_classes=self.num_classes,
        ).to(self.device)

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["ERM", f"{count_num_parameters(self.model):,}"],
        ]
        print(tabulate(model_parameters_table))

        self.optimizer = build_optimizer(self.model, self.cfg.OPTIM)
        self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)
        self.model_registeration("erm", self.model, self.optimizer, self.scheduler)

    def forward_backward(self, batch_data):
        input_data, class_label = self.parse_batch_train(batch_data)
        output = self.model(input_data)
        loss = F.cross_entropy(output, class_label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
