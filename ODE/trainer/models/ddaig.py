import timm
import torch
from tabulate import tabulate
from torch.nn import functional as F

from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils.tools import count_num_parameters
from utils.fcn import build_network


@MODEL_REGISTRY.register()
class DDAIG(Trainer):
    """Deep Domain-Adversarial Image Generation.

    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.MODEL.DDAIG.LMDA
        self.alpha = cfg.MODEL.DDAIG.ALPHA

    def build_model(self):
        print("Building Class Classifier")
        self.class_classifier = timm.create_model(
            self.cfg.MODEL.DDAIG.BACKBONE,
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
            self.cfg.MODEL.DDAIG.BACKBONE,
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

        print("Building Domain Transformation Net")
        self.domain_transformation_net = build_network(self.cfg.MODEL.DDAIG.G_ARCH).to(
            self.device
        )
        self.domain_transformation_optimizer = build_optimizer(
            self.domain_transformation_net, self.cfg.OPTIM
        )
        self.domain_transformation_scheduler = build_lr_scheduler(
            self.domain_transformation_optimizer, self.cfg.OPTIM
        )
        self.model_registeration(
            "domain_transformation_net",
            self.domain_transformation_net,
            self.domain_transformation_optimizer,
            self.domain_transformation_scheduler,
        )

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Class Classifier", f"{count_num_parameters(self.class_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"],
            [
                "Domain Transformation Net",
                f"{count_num_parameters(self.domain_transformation_net):,}",
            ],
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        #############
        # Update Domain Transformation Net
        #############
        input_data_augmented = self.domain_transformation_net(
            input_data, lmda=self.lmda
        )
        loss_domain_transformation = 0
        # Minimize Class Classifier Loss
        loss_domain_transformation += F.cross_entropy(
            self.class_classifier(input_data_augmented), class_label
        )
        # Maximize Domain Classifier Loss
        loss_domain_transformation -= F.cross_entropy(
            self.domain_classifier(input_data_augmented), domain_label
        )
        self.model_backward_and_update(
            loss_domain_transformation, "domain_transformation_net"
        )

        # Augment Data with Updated Domain Transformation Net
        with torch.no_grad():
            input_data_augmented = self.domain_transformation_net(
                input_data, lmda=self.lmda
            )

        #############
        # Update Class Classifier
        #############
        loss_class = F.cross_entropy(self.class_classifier(input_data), class_label)
        loss_class_augmented = F.cross_entropy(
            self.class_classifier(input_data_augmented), class_label
        )
        loss_class = (1 - self.alpha) * loss_class + self.alpha * loss_class_augmented
        self.model_backward_and_update(loss_class, "class_classifier")

        #############
        # Update Domain Classifier
        #############
        loss_domain = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        self.model_backward_and_update(loss_domain, "domain_classifier")

        loss_summary = {
            "loss_domain_transformation": loss_domain_transformation.item(),
            "loss_class": loss_class.item(),
            "loss_domain": loss_domain.item(),
        }

        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.class_classifier(input_data)
