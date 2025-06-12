
import argparse
from utils import set_random_seed
from clip import clip0
from clip.maple import CustomCLIP, load_clip_to_cpu, ZSCLIPModel
import os
import torch
from torch.utils.data import DataLoader
from model.meta_4 import MetaLearner #Reptile
from model.tpt0 import TPT
from transformers import get_scheduler
from datasets import MetaOfficeHomeSptReptileAug, MetaDomainNetSptReptileAug
from PIL import Image
from data.datautils import AugMixAugmenter
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import sys

build_dataset_dict = {
    "officehome": MetaOfficeHomeSptReptileAug,
    "domainnet": MetaDomainNetSptReptileAug
}

#reptile
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3, metavar='N', help='fix random seed')  
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    parser.add_argument('--data_root', type=str, default='')

    parser.add_argument('--dataname', default="officehome", type=str) 
    parser.add_argument('--source_domain', type=str, default="c/p/r")
    parser.add_argument('--target_domain', type=str, default="a") 
    
    parser.add_argument('--num_shots', default=3, type=int)# 3 6

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--inner_steps',default=3, type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--n_ctx', default=2, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--prompt_depth', default=1, type=int)

    parser.add_argument('--meta_lr',default=5e-4, type=float)
    parser.add_argument('--inner_lr',default=5e-3,type=float)

    #ttt
    parser.add_argument('--batch_ttt', default=64, type=int)
    parser.add_argument('--lr_ttt', default=5e-3, type=float) 
    parser.add_argument('--tta_steps',default=1,type=int)
    parser.add_argument('--selection_p',default=0.1, type=float)
    parser.add_argument('--print_freq', default=200, type=int)

    parser.add_argument('--output_dir', default="", type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Prepare configs and logs
    set_random_seed(args.seed)
    source_domain_list = args.source_domain.split('/')
    domain = ""
    for s in source_domain_list:
        domain += s
    domain += f"_{args.target_domain}"

    output_dir = args.output_dir + f"/{args.dataname}/inner_step{args.inner_steps}_lr{args.meta_lr}_{args.inner_lr}/"

    if args.backbone == "ViT-B/16":
        output_dir += f"vit_b16/{domain}/spt{args.num_shots}/epoch{args.epochs}" \
                      f"/tta_steps{args.tta_steps}/{args.seed}/"
    elif args.backbone == "ViT-B/32":
        output_dir += f"vit_b32/{domain}/spt{args.num_shots}/epoch{args.epochs}" \
                      f"/tta_steps{args.tta_steps}/{args.seed}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file_path = output_dir + 'log.txt'
    log_file = open(log_file_path, 'w')
  
    sys.stdout = log_file
    print(args)

    clip_model, _ = load_clip_to_cpu(args)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    meta_dataset = build_dataset_dict[args.dataname](args, transform, "train")

    model = CustomCLIP(args, classnames=meta_dataset.classnames, clip_model=clip_model)
    model.to(device)


    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    total_model_parameters = sum(p.numel() for p in model.prompt_learner.parameters())
    print(f"Total_model_parameters: {total_model_parameters/1e6}M")

    max_train_steps = args.epochs 

    optimizer_model = torch.optim.AdamW(model.prompt_learner.parameters(), lr=args.meta_lr, weight_decay=1e-2)

    lr_scheduler_model = get_scheduler(
        name="cosine",
        optimizer=optimizer_model,
        num_warmup_steps=1,
        num_training_steps=max_train_steps,
    )

    metalearner = MetaLearner(args=args, model=model, optimizer_model=optimizer_model, lr_scheduler_model=lr_scheduler_model, device=device)

    metalearner.train(meta_dataset)
    prompt_save_path = output_dir + f'prompt_{args.epochs}.pt'
    metalearner.save_model(prompt_save_path)
    del meta_dataset, metalearner

    print("========== Begining TTT for target domain! ==========")
    torch.cuda.empty_cache()
  
    meta_prompt_weight = torch.load(prompt_save_path, weights_only=False)
    with torch.no_grad():
        model.prompt_learner.load_state_dict(meta_prompt_weight, strict=False)
        model.prompt_learner.set_prompt_init_states()

    model.to(device)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_ttt - 1)

    target_dataset = build_dataset_dict[args.dataname](args, data_transform, "test")
    target_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True,
                        num_workers=12, pin_memory=True)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), args.lr_ttt)

    scaler = torch.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    tpt = TPT(args=args, model=model, optimizer=optimizer, scaler=scaler, device=device)

    results = tpt.test_time_train(target_dataloader)
    print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(args.target_domain, results[0], results[1]))

if __name__ == "__main__":
    main()