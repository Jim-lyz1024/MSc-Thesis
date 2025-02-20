import os

import torch
import torch.nn as nn
from clip import clip
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES


class Adapter(nn.Module):
    def __init__(self, channel_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        if self.cfg.MODEL.BOKD.BACKBONE == "RN50":
            adapter_dim = 1024
        elif self.cfg.MODEL.BOKD.BACKBONE == "ViT-B/32":
            adapter_dim = 512
        self.domain_aware_adapters = nn.ModuleList(
            [
                Adapter(adapter_dim, 4).to(clip_model.dtype)
                for i in range(len(cfg.DATASET.SOURCE_DOMAINS))
            ]
        )
        self.dtype = clip_model.dtype

        prompt_template = PROMPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = {}
        prompts_classic = [
            prompt_template.format(class_name.replace("_", " "))
            for class_name in class_names
        ]
        prompts["classic"] = prompts_classic
        for domain in cfg.DATASET.SOURCE_DOMAINS:
            prompts[domain] = [
                prompt_template.format(
                    domain.replace("_", " ") + " " + class_name.replace("_", " ")
                )
                for class_name in class_names
            ]

        self.text_features = {}
        for domain, prompts_domain in prompts.items():
            prompts_domain_tokenized = torch.cat(
                [clip.tokenize(prompt) for prompt in prompts_domain]
            )
            prompts_domain_tokenized = prompts_domain_tokenized.to(
                torch.cuda.current_device()
            )
            with torch.no_grad():
                self.text_features[domain] = clip_model.encode_text(
                    prompts_domain_tokenized
                )
                self.text_features[domain] = self.text_features[
                    domain
                ] / self.text_features[domain].norm(dim=-1, keepdim=True)

    def forward(self, image, domain_labels=None):
        adapter_ratio = 0.2
        image_features = self.image_encoder(image.type(self.dtype))

        adapter_features = []
        for image_feature, domain_label in zip(image_features, domain_labels):
            adapter_features.append(
                self.domain_aware_adapters[domain_label](image_feature)
            )
        adapter_features = torch.vstack(adapter_features)

        image_features = (
            adapter_ratio * adapter_features + (1 - adapter_ratio) * image_features
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_domain = {}
        for domain, text_feature in self.text_features.items():
            logits_domain[domain] = logit_scale * image_features @ text_feature.t()
        logits = torch.cat(list(logits_domain.values()), dim=1)

        return logits


@MODEL_REGISTRY.register()
class BOKD(Trainer):
    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.BOKD.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.BOKD.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        print("Building Teacher Model")
        self.teacher = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.teacher.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.teacher.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.teacher.to(self.device)

        # NOTE: Only Give domain_aware_adapters to the Optimizer
        self.optimizer_teacher = build_optimizer(
            self.teacher.domain_aware_adapters, self.cfg.OPTIM
        )
        self.lr_scheduler_teacher = build_lr_scheduler(
            self.optimizer_teacher, self.cfg.OPTIM
        )

        self.model_registeration(
            "BOKD_Teacher",
            self.teacher.domain_aware_adapters,
            self.optimizer_teacher,
            self.lr_scheduler_teacher,
        )

    def forward_backward(self, batch_data):
        image, class_label, domain_label = self.parse_batch_train(batch_data)
        logits_teacher = torch.split(
            self.teacher(image, domain_label), self.num_classes, dim=1
        )

        loss_teacher = F.cross_entropy(logits_teacher[0], class_label, reduction="none")

        for i in range(len(domain_label)):
            for idx, domain_output in enumerate(logits_teacher[1:]):
                if domain_label[i] == idx:
                    domain_loss = F.cross_entropy(domain_output, class_label)
                else:
                    domain_loss = -0.1 * F.cross_entropy(domain_output, class_label)
                loss_teacher[i] = loss_teacher[i] + domain_loss
        loss_teacher = loss_teacher.mean()
        self.model_backward_and_update(loss_teacher, model_names="BOKD_Teacher")

        loss_summary = {
            "loss_teacher": loss_teacher.item(),
            "acc_teacher": compute_accuracy(logits_teacher, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        print(loss_summary)
        exit()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label
