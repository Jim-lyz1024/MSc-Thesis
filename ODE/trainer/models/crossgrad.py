import timm
import torch
from tabulate import tabulate
from torch.nn import functional as F

from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils.tools import count_num_parameters


@MODEL_REGISTRY.register()
class CrossGrad(Trainer):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_c = cfg.MODEL.CrossGrad.EPS_C
        self.eps_d = cfg.MODEL.CrossGrad.EPS_D
        self.alpha_c = cfg.MODEL.CrossGrad.ALPHA_C
        self.alpha_d = cfg.MODEL.CrossGrad.ALPHA_D

    def build_model(self):
        print("Building Class Classifier")
        self.class_classifier = timm.create_model(
            self.cfg.MODEL.CrossGrad.BACKBONE,
            pretrained=True,
            num_classes=self.num_classes,
        ).to(self.device)
        self.class_optimizer = build_optimizer(self.class_classifier, self.cfg.OPTIM)
        self.class_scheduler = build_lr_scheduler(self.class_optimizer, self.cfg.OPTIM)
        self.model_registeration(
            "class_classifier",
            self.class_classifier,
            self.class_optimizer,
            self.class_scheduler,
        )

        print("Building Domain Classifier")
        self.domain_classifier = timm.create_model(
            self.cfg.MODEL.CrossGrad.BACKBONE,
            pretrained=True,
            num_classes=self.num_source_domains,
        ).to(self.device)
        self.domain_optimizer = build_optimizer(self.domain_classifier, self.cfg.OPTIM)
        self.domain_scheduler = build_lr_scheduler(
            self.domain_optimizer, self.cfg.OPTIM
        )
        self.model_registeration(
            "domain_classifier",
            self.domain_classifier,
            self.domain_optimizer,
            self.domain_scheduler,
        )

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Class Classifier", f"{count_num_parameters(self.class_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"],
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        # Compute Domain Perturbated Data
        loss_domain = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        loss_domain.backward()
        grad_domain = torch.clamp(input_data.grad.data, min=-0.1, max=0.1)
        input_data_domain = input_data.data + self.eps_c * grad_domain

        # Compute Class Perturbated Data
        input_data.grad.data.zero_()
        loss_class = F.cross_entropy(self.class_classifier(input_data), class_label)
        loss_class.backward()
        grad_class = torch.clamp(input_data.grad.data, min=-0.1, max=0.1)
        input_data_class = input_data.data + self.eps_d * grad_class

        input_data = input_data.detach()

        # Update Class Classifier
        loss_class = F.cross_entropy(self.class_classifier(input_data), class_label)
        loss_class_domain = F.cross_entropy(
            self.class_classifier(input_data_domain), class_label
        )
        loss_class = (1 - self.alpha_c) * loss_class + self.alpha_c * loss_class_domain
        self.model_backward_and_update(loss_class, "class_classifier")

        # Update Domain Classifier
        loss_domain = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        loss_domain_class = F.cross_entropy(
            self.domain_classifier(input_data_class), domain_label
        )
        loss_domain = (
            1 - self.alpha_d
        ) * loss_domain + self.alpha_d * loss_domain_class
        self.model_backward_and_update(loss_domain, "domain_classifier")

        loss_summary = {
            "loss_class": loss_class.item(),
            "loss_domain": loss_domain.item(),
        }

        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        input_data.requires_grad = True
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.class_classifier(input_data)
