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
        if self.cfg.MODEL.DODGE.BACKBONE == "RN50":
            adapter_dim = 1024
        elif self.cfg.MODEL.DODGE.BACKBONE == "ViT-B/32":
            adapter_dim = 512
        self.adapters = nn.ModuleList(
            [
                Adapter(adapter_dim, 4).to(clip_model.dtype)
                for i in range(len(cfg.DATASET.SOURCE_DOMAINS))
            ]
        )
        self.dtype = clip_model.dtype
        self.class_names = class_names
        self.clip_model = clip_model
        
        self.text_features = {}
        self.text_features2 = {}
        self.sim_scores = []
        self.initialize_text_features()
    
    def initialize_text_features(self):
        def generate_prompts(domain_names, prompt_template):
            prompts_domain = {}
            prompts_original = [prompt_template.format(class_name.replace("_", " ")) for class_name in self.class_names]
            prompts_domain['original'] = prompts_original

            for domain in domain_names:
                prompts_domain[domain] = [
                    prompt_template.format(domain.replace("_", " ") + ' ' + class_name.replace("_", " "))
                    for class_name in self.class_names
                ]
            return prompts_domain

        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        
        source_domain_names = self.cfg.DATASET.SOURCE_DOMAINS
        target_domain_names = self.cfg.DATASET.TARGET_DOMAINS

        # Generate prompts for source and target domains
        source_prompts = generate_prompts(source_domain_names, prompt_template)
        target_prompts = generate_prompts(target_domain_names, prompt_template)

        print(source_prompts)
        print(target_prompts)

        def encode_text(prompts):
            text_features = {}
            for domain, prompts_list in prompts.items():
                tokenized_prompts = [clip.tokenize(prompt) for prompt in prompts_list]
                tokenized_prompts = torch.cat(tokenized_prompts).to(torch.cuda.current_device())
                with torch.no_grad():
                    text_features[domain] = self.clip_model.encode_text(tokenized_prompts)
                    text_features[domain] = text_features[domain] / text_features[domain].norm(dim=-1, keepdim=True)
            return text_features

        self.text_features = encode_text(source_prompts)
        self.text_features2 = encode_text(target_prompts)

        tar_f = self.text_features2[target_domain_names[0]]
        sim_scores = [F.cosine_similarity(v.flatten(), tar_f.flatten(), dim=0) for v in self.text_features.values()]
        self.sim_scores = sim_scores[1:]
        
    def forward(self, image, domain_labels=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        adapter_ratio = 0.2
        # Training phase
        if domain_labels is not None:
            adapter_features = []
            for image_feature, domain_label in zip(image_features, domain_labels):
                adapter_features.append(self.adapters[domain_label](image_feature))
            adapter_features = torch.vstack(adapter_features)
            # Use fixed adapter_ratio during training
            image_features = (adapter_ratio * adapter_features + (1 - adapter_ratio) * image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits using all prompts
            logit_scale = self.logit_scale.exp()
            logits_domain = {}
            for domain, text_feature in self.text_features.items():
                logits_domain[domain] = logit_scale * image_features @ text_feature.t()
            logits = torch.cat(list(logits_domain.values()), dim=1)
        # Inference phase
        else:
            adapter_features_list = []
            for adapter in self.adapters:
                adapter_feature = adapter(image_features)
                adapter_features_list.append(adapter_feature)
            # Compute weights using softmax
            sim_scores_tensor = torch.tensor(self.sim_scores).to(self.dtype).to(image_features.device)
            weights = F.softmax(sim_scores_tensor, dim=0)
            combined_adapter_features = sum(w * f for w, f in zip(weights, adapter_features_list))
            # print("Weights: ", weights)

            # Compute balancing parameter beta
            sum_sim = sim_scores_tensor.sum()
            # print("Sum Similarity: ", sum_sim)
            max_sim_sum = len(self.sim_scores)  # Since max similarity is 1 for each
            beta = sum_sim / max_sim_sum
            beta = beta.item()
            # print("Beta: ", beta)
            
            combined_adapter_features = beta * combined_adapter_features

            # Combine adapter features and image features using beta
            image_features = (adapter_ratio * combined_adapter_features + (1 - adapter_ratio) * image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits using only 'original' prompts
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features['original'].t()

        return logits
    
    
@MODEL_REGISTRY.register()
class DODGE(Trainer):
    def build_model(self):
        # domain_names = self.data_manager.dataset.domains
        # print("Domain: ", domain_names)
    
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.DODGE.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.DODGE.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        print("Building DODGE Model")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # NOTE: Only Give domain_aware_adapters to the Optimizer
        self.optimizer = build_optimizer(self.model.adapters, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        # Encapsulate the three operations of optimize, simplify the process by just use the model, don't have to perform the three operations in turn
        self.model_registeration(
            "dodge",
            self.model.adapters,
            self.optimizer,
            self.lr_scheduler,
        )

    def forward_backward(self, batch_data):
        image, class_label, domain_label = self.parse_batch_train(batch_data)
        all_domains = self.model(image, domain_labels=domain_label)
        domains_outputs = torch.split(all_domains, self.num_classes, dim=1)

        loss_by_domain = F.cross_entropy(domains_outputs[0], class_label, reduction="none")
                
        for i in range(len(domain_label)):
            for idx, domain_output in enumerate(domains_outputs[1:]):
                if domain_label[i] == idx:
                    domain_loss = F.cross_entropy(domain_output, class_label)
                else:
                    domain_loss = -0.1 * F.cross_entropy(domain_output, class_label)
                loss_by_domain[i] = loss_by_domain[i] + domain_loss
        loss_by_domain = loss_by_domain.mean()
        self.model_backward_and_update(loss_by_domain, model_names="dodge")

        loss_summary = {
            "loss": loss_by_domain.item(),
            "acc": compute_accuracy(domains_outputs, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        # print(loss_summary)
        # exit()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label