import timm
import torch
from tabulate import tabulate
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils.tools import count_num_parameters


@MODEL_REGISTRY.register()
class DomainMix(Trainer):
    """DomainMix.

    Dynamic Domain Generalization.

    https://github.com/MetaVisionLab/DDG
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mix_type = cfg.MODEL.DomainMix.TYPE
        self.alpha = cfg.MODEL.DomainMix.ALPHA
        self.beta = cfg.MODEL.DomainMix.BETA
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def build_model(self):
        self.model = timm.create_model(
            self.cfg.MODEL.DomainMix.BACKBONE,
            pretrained=True,
            num_classes=self.num_classes,
        ).to(self.device)

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["DomainMix", f"{count_num_parameters(self.model):,}"],
        ]
        print(tabulate(model_parameters_table))

        self.optimizer = build_optimizer(self.model, self.cfg.OPTIM)
        self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)
        self.model_registeration("domainmix", self.model, self.optimizer, self.scheduler)

    def forward_backward(self, batch_data):
        input_data, label_a, label_b, lam = self.parse_batch_train(batch_data)
        output = self.model(input_data)

        loss_class_a = F.cross_entropy(output, label_a)
        loss_class_b = F.cross_entropy(output, label_b)
        loss_class = lam * loss_class_a + (1 - lam) * loss_class_b

        self.model_backward_and_update(loss_class)

        loss_summary = {
            "loss": loss_class.item(),
            "acc": compute_accuracy(output, label_a)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)

        input_data, label_a, label_b, lam = self.domain_mix(
            input_data, class_label, domain_label
        )

        return input_data, label_a, label_b, lam

    def domain_mix(self, input_data, class_label, domain_label):
        if self.alpha > 0:
            lam = self.dist_beta.rsample((1,)).to(input_data.device)
        else:
            lam = torch.tensor(1).to(input_data.device)

        perm = torch.randperm(
            input_data.size(0), dtype=torch.int64, device=input_data.device
        )

        if self.mix_type == "crossdomain":
            domain_list = torch.unique(domain_label)
            if len(domain_list) > 1:
                for current_domain_index in domain_list:
                    # Count the number of examples in the current domain.
                    count_current_domain = torch.sum(
                        domain_label == current_domain_index
                    )
                    # Retrieve the index of examples other than the current domain.
                    other_domain_index = (
                        (domain_label != current_domain_index).nonzero().squeeze(-1)
                    )
                    # Count the number of examples in the other domains.
                    count_other_domain = other_domain_index.shape[0]
                    # Generate the Perm index of examples in the other domains that are going to be mixed.
                    perm_other_domain = torch.ones(count_other_domain).multinomial(
                        num_samples=count_current_domain,
                        replacement=bool(count_current_domain > count_other_domain),
                    )
                    # Replace current domain label with another domain label.
                    perm[domain_label == current_domain_index] = other_domain_index[
                        perm_other_domain
                    ]
        elif self.mix_type != "random":
            raise NotImplementedError(
                f"Mix Type should be within {'random', 'crossdomain'}, but got {self.mix_type}"
            )

        mixed_input_data = lam * input_data + (1 - lam) * input_data[perm, :]
        label_a, label_b = class_label, class_label[perm]
        return mixed_input_data, label_a, label_b, lam
