from copy import deepcopy
import time
import torch
import numpy as np
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy

class TPT():
    def __init__(self, args, model, optimizer, scaler, device):
        self.args = args
        self.model = model
       
        self.optimizer = optimizer
        self.optim_state = deepcopy(optimizer.state_dict())
        self.scaler = scaler
        self.device = device
        self.tta_steps = args.tta_steps
        self.selection_p = args.selection_p

    def select_confident_samples(self, logits, top):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
        return logits[idx], idx

    def avg_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

    def test_time_tuning(self, inputs):
        selected_idx = None
        for j in range(self.tta_steps):
            with torch.amp.autocast('cuda'):
                output, image_feature = self.model(inputs, return_img=True)

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = self.select_confident_samples(output, self.selection_p)

                loss = self.avg_entropy(output)

            self.optimizer.zero_grad()
          
            self.scaler.scale(loss).backward()
           
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return

    def test_time_train(self, target_dataloader):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(target_dataloader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # reset model and switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            self.model.reset()
        end = time.time()
        for i, (images, target, classname) in enumerate(target_dataloader):
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].to(self.device) 
                image = images[0]
            else:
                if len(images.size()) > 4:
                  
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.to(self.device)
                image = images
            target = target.to(self.device) 

            images = torch.cat(images, dim=0)
           
            if self.args.tta_steps > 0:
                with torch.no_grad():
                    self.model.reset()

            self.optimizer.load_state_dict(self.optim_state)

            self.test_time_tuning(images)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output = self.model(image)
          
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]