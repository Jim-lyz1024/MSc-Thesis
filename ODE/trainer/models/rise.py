import os

import timm
import torch
import torch.nn as nn
from clip import clip
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES


@MODEL_REGISTRY.register()
class RISE(Trainer):
    """RISE

    A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance
    https://arxiv.org/abs/2309.12530
    """

    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.RISE.BACKBONE))
        self.clip_model, _ = clip.load(
            self.cfg.MODEL.RISE.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )
        if self.cfg.MODEL.RISE.BACKBONE == "RN50":
            self.text_feature_dim = 1024
        elif self.cfg.MODEL.RISE.BACKBONE == "ViT-B/32":
            self.text_feature_dim = 512

        # Construct Prompts
        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [
            prompt_template.format(class_name.replace("_", " "))
            for class_name in self.data_manager.dataset.class_names
        ]
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            self.clip_model.eval()
            self.text_features = self.clip_model.encode_text(prompts)
            self.text_features_norm = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

        # Construct Prompts with domain. Used to compute relative distance loss.
        prompts_domain = []
        for source_domain in self.cfg.DATASET.SOURCE_DOMAINS:
            prompts_domain.append(
                torch.cat(
                    [
                        clip.tokenize("a {} of a {}.".format(source_domain, class_name))
                        for class_name in self.data_manager.dataset.class_names
                    ]
                ).to(self.device)
            )
        text_features_domain = []
        with torch.no_grad():
            for prompt_domain in prompts_domain:
                text_features_domain.append(self.clip_model.encode_text(prompt_domain))
        self.text_feature_domains = torch.zeros(
            self.num_classes,
            len(self.cfg.DATASET.SOURCE_DOMAINS),
            self.text_feature_dim,
        ).to(self.device)
        for i in range(self.num_classes):
            for j in range(len(self.cfg.DATASET.SOURCE_DOMAINS)):
                self.text_feature_domains[i, j, :] = text_features_domain[j][i]

        self.student_model = timm.create_model(
            self.cfg.MODEL.RISE.STUDENT_NETWORK,
            pretrained=True,
            num_classes=self.num_classes,
        )

        teacher_network = self.cfg.MODEL.RISE.BACKBONE
        student_network = self.cfg.MODEL.RISE.STUDENT_NETWORK

        if not (teacher_network == "ViT-B/32" and student_network == "resnet18"):
            if teacher_network == "ViT-B/32":
                self.student_model.projection_layer = nn.Linear(
                    self.student_model.fc.in_features, 512, bias=True
                )
            elif teacher_network == "RN50":
                self.student_model.projection_layer = nn.Linear(
                    self.student_model.fc.in_features, 1024, bias=True
                )
            else:
                raise NotImplementedError

            del self.student_model.fc
            self.student_model.fc = nn.Linear(
                self.student_model.projection_layer.out_features,
                self.num_classes,
                bias=True,
            )
        self.student_model.to(self.device)

        self.optimizer = build_optimizer(self.student_model, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)
        self.model_registeration(
            "rise",
            self.student_model,
            self.optimizer,
            self.lr_scheduler,
        )

        self.distillation_loss_weight = self.cfg.MODEL.RISE.LOSS_WEIGHT.DISTILLATION
        self.classification_loss_weight = self.cfg.MODEL.RISE.LOSS_WEIGHT.CLASSIFICATION
        self.distance_loss_weight = self.cfg.MODEL.RISE.LOSS_WEIGHT.DISTANCE
        self.temperature = self.cfg.MODEL.RISE.TEMPERATURE

        print("Distillation_Loss_Weight: {}".format(self.distillation_loss_weight))
        print("Classification_Loss_Weight: {}".format(self.classification_loss_weight))
        print("Distance Loss Weight: {}".format(self.distance_loss_weight))
        print("Temperature: {}".format(self.temperature))

    def forward_backward(self, batch_data):
        image, class_label = self.parse_batch_train(batch_data)

        # Compute Image Features for both Teacher (CLIP) and Student
        with torch.no_grad():
            self.clip_model.eval()
            teacher_image_features = self.clip_model.encode_image(image)

        teacher_image_features = teacher_image_features / teacher_image_features.norm(
            dim=-1, keepdim=True
        )
        logit_scale = self.clip_model.logit_scale.exp()
        teacher_logits = (
            logit_scale * teacher_image_features @ self.text_features_norm.T
        )
        student_image_features = self.student_model.forward_features(image)
        student_image_features = self.student_model.global_pool(student_image_features)
        if not (
            self.cfg.MODEL.RISE.BACKBONE == "ViT-B/32"
            and self.cfg.MODEL.RISE.STUDENT_NETWORK == "resnet18"
        ):
            student_image_features = self.student_model.projection_layer(
                student_image_features
            )

        student_logits = self.student_model.fc(student_image_features)

        # --- Classification Loss
        classification_loss = F.cross_entropy(student_logits, class_label)

        # --- Distillation Loss
        distillation_loss = (
            F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=1),
                F.softmax(teacher_logits / self.temperature, dim=1),
                reduction="batchmean",
            )
            * self.temperature
            * self.temperature
        )

        # --- Absolute Distance Loss
        teacher_text_features = torch.zeros(
            student_image_features.shape[0], self.text_feature_dim
        ).to(self.device)
        for i in range(image.shape[0]):
            teacher_text_features[i, :] = self.text_features_norm[class_label[i], :]
        absolute_distance_loss = F.cosine_embedding_loss(
            F.normalize(student_image_features, dim=-1),
            teacher_text_features,
            torch.ones(student_image_features.shape[0]).to(self.device),
        )

        # --- Relative Distance Loss
        distance_teacher = torch.zeros(
            image.shape[0], self.data_manager.num_source_domains
        ).to(self.device)
        distance_student = torch.zeros(
            image.shape[0], self.data_manager.num_source_domains
        ).to(self.device)

        for i in range(image.shape[0]):
            student_image_feature = student_image_features[i, :]
            ground_truth_class_label = class_label[i]
            teacher_text_feature = self.text_features[ground_truth_class_label]
            anchor_text_feature = self.text_feature_domains[ground_truth_class_label]

            distance_teacher[i, :] = F.cosine_similarity(
                teacher_text_feature.repeat(self.data_manager.num_source_domains, 1),
                anchor_text_feature,
            )
            distance_student[i, :] = F.cosine_similarity(
                student_image_feature.repeat(self.data_manager.num_source_domains, 1),
                anchor_text_feature,
            )

        distance_teacher = F.softmax(distance_teacher, dim=1)
        distance_student = F.softmax(distance_student, dim=1)
        relative_distance_loss = F.mse_loss(distance_student, distance_teacher) * 10.0

        loss = (
            distillation_loss * self.distillation_loss_weight
            + classification_loss * self.classification_loss_weight
            + absolute_distance_loss * self.distance_loss_weight
            + relative_distance_loss * self.distance_loss_weight
        )

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "distil_loss": distillation_loss.item(),
            "class_loss": classification_loss.item(),
            "absolute_loss": absolute_distance_loss.item(),
            "relative_loss": relative_distance_loss.item(),
            "acc": compute_accuracy(student_logits, class_label)[0].item(),
        }

        return loss_summary

    def model_inference(self, input_data):
        if (
            self.cfg.MODEL.RISE.BACKBONE == "ViT-B/32"
            and self.cfg.MODEL.RISE.STUDENT_NETWORK == "resnet18"
        ):
            return self.student_model(input_data)
        else:
            image_features = self.student_model.forward_features(input_data)
            image_features = self.student_model.global_pool(image_features)
            image_features = self.student_model.projection_layer(image_features)
            return self.student_model.fc(image_features)
