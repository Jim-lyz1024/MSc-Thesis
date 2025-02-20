import copy

import timm
import torch
from tabulate import tabulate
from torch.nn import functional as F
from tqdm import tqdm

import wandb  # noqa

# from ops import compute_perturbation_weight, measure_diversity
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils.fcn import build_network
from utils.tools import count_num_parameters


@MODEL_REGISTRY.register()
class DAF(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.alpha = 0.5
        self.lmda = 0.3

        self.best_accuracy = 0

    def build_model(self):
        print("Building Class Classifier")
        self.class_classifier = timm.create_model(
            self.cfg.MODEL.DAF.BACKBONE,
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

        print("Building DoTNet - Domain Generator")
        self.domain_generator = build_network(self.cfg.MODEL.DAF.G_ARCH).to(self.device)
        self.domain_generator_optimizer = build_optimizer(
            self.domain_generator, self.cfg.OPTIM
        )
        self.domain_generator_scheduler = build_lr_scheduler(
            self.domain_generator_optimizer, self.cfg.OPTIM
        )
        self.model_registeration(
            "domain_generator",
            self.domain_generator,
            self.domain_generator_optimizer,
            self.domain_generator_scheduler,
        )
        print("Building DoTNet - Domain Discriminator")
        self.domain_discriminator = timm.create_model(
            self.cfg.MODEL.DAF.BACKBONE,
            pretrained=True,
            num_classes=self.num_source_domains,
        ).to(self.device)
        self.domain_discriminator_optimizer = build_optimizer(
            self.domain_discriminator, self.cfg.OPTIM
        )
        self.domain_discriminator_scheduler = build_lr_scheduler(
            self.domain_discriminator_optimizer, self.cfg.OPTIM
        )
        self.model_registeration(
            "domain_discriminator",
            self.domain_discriminator,
            self.domain_discriminator_optimizer,
            self.domain_discriminator_scheduler,
        )

        print("Building ACTNet - Perturbation Generator")
        self.perturbation_generator = build_network(self.cfg.MODEL.DAF.G_ARCH).to(
            self.device
        )
        self.perturbation_generator_optimizer = build_optimizer(
            self.perturbation_generator, self.cfg.OPTIM
        )
        self.perturbation_generator_scheduler = build_lr_scheduler(
            self.perturbation_generator_optimizer, self.cfg.OPTIM
        )
        self.model_registeration(
            "perturbation_generator",
            self.perturbation_generator,
            self.perturbation_generator_optimizer,
            self.perturbation_generator_scheduler,
        )
        print("Building ACTNet - Class Discriminator")
        self.class_discriminator = timm.create_model(
            self.cfg.MODEL.DAF.BACKBONE,
            pretrained=True,
            num_classes=self.num_classes,
        ).to(self.device)
        self.class_discriminator_optimizer = build_optimizer(
            self.class_discriminator, self.cfg.OPTIM
        )
        self.class_discriminator_scheduler = build_lr_scheduler(
            self.class_discriminator_optimizer, self.cfg.OPTIM
        )
        self.model_registeration(
            "class_discriminator",
            self.class_discriminator,
            self.class_discriminator_optimizer,
            self.class_discriminator_scheduler,
        )

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Class Classifier", f"{count_num_parameters(self.class_classifier):,}"],
            [
                "DoTNet - Domain Generator",
                f"{count_num_parameters(self.domain_generator):,}",
            ],
            [
                "DoTNet - Domain Discriminator",
                f"{count_num_parameters(self.domain_discriminator):,}",
            ],
            [
                "ACTNet - Perturbation Generator",
                f"{count_num_parameters(self.perturbation_generator):,}",
            ],
            [
                "ACTNet - Class Discriminator",
                f"{count_num_parameters(self.class_discriminator):,}",
            ],
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        # if self.current_epoch + 1 <= 5:
        #     loss_class_classifier = F.cross_entropy(
        #         self.class_classifier(input_data), class_label
        #     )
        #     self.model_backward_and_update(loss_class_classifier, "class_classifier")
        #     loss_summary = {"loss_class_classifier": loss_class_classifier.item()}

        #     if self.batch_idx + 1 == self.num_batches:
        #         self.update_lr()

        #     return loss_summary

        # Compute Diversity and Dynamic Lambda
        # with torch.no_grad():
        #     embeddings = self.class_classifier.forward_features(input_data)
        #     embeddings_normalized = (embeddings - torch.min(embeddings)) / (
        #         torch.max(embeddings) - torch.min(embeddings)
        #     )
        #     embeddings_diversity = measure_diversity(
        #         embeddings_normalized.cpu(), diversity_type="gini"
        #     )
        #     self.lmda = compute_perturbation_weight(embeddings_diversity)

        ############
        # Update DoTNet - Domain Generator Net
        ############
        input_data_domain_augmented = self.domain_generator(input_data, lmda=self.lmda)
        loss_domain_generator = F.cross_entropy(
            self.class_classifier(input_data_domain_augmented), class_label
        )
        loss_domain_generator -= F.cross_entropy(
            self.domain_discriminator(input_data_domain_augmented), domain_label
        )
        self.model_backward_and_update(loss_domain_generator, "domain_generator")

        # Augment Data with Updated Domain Generator
        with torch.no_grad():
            input_data_domain_augmented = self.domain_generator(
                input_data, lmda=self.lmda
            )

        #############
        # Update ACTNet - Perturbation Generator
        #############
        input_data_class_augmented = self.perturbation_generator(
            input_data, lmda=self.lmda
        )
        loss_perturbation_generator = F.cross_entropy(
            self.class_classifier(input_data), class_label
        )
        loss_perturbation_generator -= F.cross_entropy(
            self.class_discriminator(input_data_class_augmented), class_label
        )
        self.model_backward_and_update(
            loss_perturbation_generator, "perturbation_generator"
        )

        # Augment Data with Updated Perturbation Generator
        with torch.no_grad():
            input_data_class_augmented = self.perturbation_generator(
                input_data, lmda=self.lmda
            )

        #############
        # Update Class Classifier
        #############
        loss_class_classifier = F.cross_entropy(
            self.class_classifier(input_data), class_label
        )
        loss_class_classifier += F.cross_entropy(
            self.class_classifier(input_data_domain_augmented), class_label
        )
        loss_class_classifier += F.cross_entropy(
            self.class_classifier(input_data_class_augmented), class_label
        )
        self.model_backward_and_update(loss_class_classifier, "class_classifier")

        #############
        # Update DoTNet - Domain Discriminator
        #############
        loss_domain_discriminator = F.cross_entropy(
            self.domain_discriminator(input_data), domain_label
        )
        self.model_backward_and_update(
            loss_domain_discriminator, "domain_discriminator"
        )

        #############
        # Update ACTNet - Class Discriminator
        #############
        loss_class_discriminator = F.cross_entropy(
            self.class_discriminator(input_data), class_label
        )
        loss_class_discriminator += F.cross_entropy(
            self.class_discriminator(input_data_class_augmented), class_label
        )

        loss_summary = {
            "loss_class_classifier": loss_class_classifier.item(),
            "loss_domain_generator": loss_domain_generator.item(),
            "loss_domain_discriminator": loss_domain_discriminator.item(),
            "loss_perturbation_generator": loss_perturbation_generator.item(),
            "loss_class_discriminator": loss_class_discriminator.item(),
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

    def after_epoch(self):
        print("After Epoch Testing")
        test_class_classifier = copy.deepcopy(self.class_classifier)
        test_class_classifier.eval()

        self.evaluator.reset()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.data_loader_test)):
                input_data, class_label = self.parse_batch_test(batch_data)
                output = test_class_classifier(input_data)
                self.evaluator.process(output, class_label)
        evaluation_results = self.evaluator.evaluate()
        if self.best_accuracy < evaluation_results["accuracy"]:
            self.best_accuracy = evaluation_results["accuracy"]
        print("Best Accuracy: {}".format(self.best_accuracy))

        # wandb.log({"accuracy_train": evaluation_results["accuracy"]})
