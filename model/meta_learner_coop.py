
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.autograd as autograd
from scipy import spatial
from collections import defaultdict
import random
import math
from copy import deepcopy

class MetaLearner():
    def __init__(self, args, model, optimizer_model, lr_scheduler_model, device):
        self.args = args
        self.model = model

        self.dtype = model.prompt_learner.dtype
        self.logit_scale = model.logit_scale

        self.num_source_domain = len(args.source_domain.split("/"))
        self.device = device

        self.batch_size = args.batch_size
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.inner_steps = args.inner_steps

        self.optimizer_model = optimizer_model
        self.lr_scheduler_model = lr_scheduler_model

        self.total_epochs = self.args.epochs

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def save_model(self, prompt_path):
        prompt_state_dict = self.model.prompt_learner.state_dict()
        # Ignore fixed token vectors
        if "token_prefix" in prompt_state_dict:
            del prompt_state_dict["token_prefix"]

        if "token_suffix" in prompt_state_dict:
            del prompt_state_dict["token_suffix"]

        torch.save(prompt_state_dict, prompt_path)
        print(f"Prompt weight save in {prompt_path}")


    def prompt_clone(self):
        return self.model.prompt_learner.ctx.detach().clone()

    def compute_lambda(self):
        lamb = random.uniform(0.0, 1.0)
        print(f"lamb: {lamb:.4f}")
        return lamb

    def train(self, meta_dataset):
        num_classes = len(meta_dataset.classnames)

        for epoch in tqdm(range(self.total_epochs)):

            total_meta_loss = 0.0
            task_diff = []
            inner_loss_list = []

            theta_clone = self.prompt_clone() 
            prompt_learner_clone = deepcopy(self.model.prompt_learner)

            random.seed(epoch + self.args.seed)
            meta_dataset.creat_batch()

            self.optimizer_model.zero_grad()

            for id, support_data in enumerate(meta_dataset): 
                if support_data != None:
                    support_dataloader = DataLoader(support_data, num_workers=16, shuffle=True, batch_size=self.batch_size)
                else:
                    break

                prompt_learner_i = deepcopy(prompt_learner_clone)
                optimizer = torch.optim.AdamW(prompt_learner_i.parameters(), lr=self.inner_lr, weight_decay=1e-2) 

                self.model.train()
                lamb = self.compute_lambda()
                lamb = torch.tensor(lamb, device=self.device).unsqueeze(0).unsqueeze(1)  

                #inner loop
                meta_loss = 0.0
                for _ in range(self.inner_steps):
                    inner_loss = 0.0
                    for step, batch_spt in enumerate(support_dataloader):
                        optimizer.zero_grad()
                        logit_scale = self.logit_scale.exp()
                        text_features = self.model.get_text_features(prompt_learner_i.ctx)
                        if len(batch_spt) == 2:
                            images = batch_spt[0].to(self.device)

                            targets = batch_spt[1].to(self.device)
                            with torch.no_grad():
                                image_features = self.model.image_encoder(images.type(self.dtype))
                                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                            logits = logit_scale * image_features @ text_features.t()
                            loss = F.cross_entropy(logits, targets).to(self.device)

                        elif len(batch_spt) == 4:
                            images1 = batch_spt[0].to(self.device)  
                            targets1 = batch_spt[1].to(self.device)
                            images2 = batch_spt[2].to(self.device)
                            targets2 = batch_spt[3].to(self.device)

                            with torch.no_grad():
                                image_aug = images1 * lamb.unsqueeze(2).unsqueeze(3).expand_as(images1) + images2 * (1 - lamb).unsqueeze(2).unsqueeze(3).expand_as(images2)  
                                image_features_aug = self.model.image_encoder(image_aug.type(self.dtype))
                                image_features_aug = image_features_aug / image_features_aug.norm(dim=-1, keepdim=True)
                                onehot_label1 = F.one_hot(targets1, num_classes=num_classes)  
                                onehot_label2 = F.one_hot(targets2, num_classes=num_classes) 
                                onehot_label_aug = onehot_label1 * lamb.expand_as(onehot_label1) + onehot_label2 * (1 - lamb).expand_as(onehot_label2)

                            logits = logit_scale * image_features_aug @ text_features.t()
                            log_probs = torch.log_softmax(logits, dim=1)
                            loss = -(onehot_label_aug * log_probs).sum(dim=1).mean()  

                        loss /= len(support_dataloader)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(prompt_learner_i.parameters(), max_norm=1.0) 
                        optimizer.step()
                        inner_loss = loss.detach().float().item()

                   
                    meta_loss += inner_loss 

                inner_loss_list.append(meta_loss / self.inner_steps)
                total_meta_loss += meta_loss / self.inner_steps

                #outer loop
                with torch.no_grad(): 
                    diff = prompt_learner_i.ctx - theta_clone.data
                    diff = diff / (diff.norm() + 1e-8)
                    task_diff.append(diff)
                

            # update
            total_meta_loss /= len(meta_dataset) 
          

            avg_diff = torch.stack(task_diff).mean(dim=0)

            self.model.prompt_learner.ctx.grad = -avg_diff

            torch.nn.utils.clip_grad_norm_(self.model.prompt_learner.parameters(), max_norm=1.0) 
            self.optimizer_model.step()#


            self.lr_scheduler_model.step()
            del theta_clone, prompt_learner_clone

            print("Updated ctx:", self.model.prompt_learner.ctx.norm().item())
            print(f"Epoch: {epoch}/{self.args.epochs}, total meta loss: {total_meta_loss}") 

