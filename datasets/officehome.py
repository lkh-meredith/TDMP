import os.path as osp
import os
from dassl.data.datasets import Datum 
from torch.utils.data import Dataset
from utils.templates import DATASETS_DOMAINS
from PIL import Image
from collections import defaultdict
import random
import numpy as np
import itertools

class MetaOfficeHomeSptReptileAug(Dataset):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "OfficeHome"
    domains = DATASETS_DOMAINS[dataset_dir]["domains"] #["art", "clipart", "product", "real_world"]
    domain_ids = DATASETS_DOMAINS[dataset_dir]["domain_ids"]
    ids_domain = DATASETS_DOMAINS[dataset_dir]["ids_domain"]
    domains_short = DATASETS_DOMAINS[dataset_dir]["domains_short"]

    def __init__(self,  args, data_transform, mode="train"):
        root = osp.abspath(osp.expanduser(args.data_root))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.split_fewshot_path = os.path.join(self.dataset_dir, "splits")

        self.num_shots = args.num_shots
     
        self.seed = args.seed

        self.target_domain = self.domains_short[args.target_domain] 
        source_domain = args.source_domain.split("/")
        self.source_domain = [self.domains_short[d] for d in source_domain]  
        # print(self.source_domain)
        self.transform = data_transform
        self.mode = mode

        if mode == "train":
            self.source_support, self.source_unlabeled = self._read_data_from_txt_to_dict(self.source_domain)


            self.label2cname, self.cname2label, self.classnames = self.get_lab2cname(self.source_support[0])

        elif mode == "test":
            self.target = self._read_data_from_txt_to_dict(self.target_domain)

            self.test = self.target[0]
            self.label2cname, self.cname2label, self.classnames = self.get_lab2cname(self.target[0])

    def _read_data_from_txt(self, labeled_file_name):
        items_dict = defaultdict(list)
        if os.path.exists(labeled_file_name):
            with open(labeled_file_name, "r") as f:
                lines = f.readlines()
                for line in lines:
                    text_list = line.split()
                    image_path, label = text_list[0], text_list[1]
                    item = [image_path, label]
                    items_dict[label].append(item)

        return items_dict

    def _read_data_from_txt_to_dict(self, input_domains):
        if input_domains == self.source_domain:
            spt_items_dict = defaultdict(list)
            u_items_dict = defaultdict(list)
            for id, dname in enumerate(input_domains):
                if dname == "Real World":
                    file_dname = "Real"
                else:
                    file_dname = dname
                domain_id = self.domain_ids[dname]
                support_file = os.path.join(self.split_fewshot_path, f"{file_dname}_labeled_p0{self.num_shots}.txt")

                if os.path.exists(support_file):
                    print(f"Find the support file: {support_file}!")
                    with open(support_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_list = line.split()
                            image_path, label = text_list[0], text_list[1]
                            class_name = image_path.split('/')[1]
                            item = Datum(
                                        impath=os.path.join(self.dataset_dir,image_path),
                                        label=int(label),
                                        domain=domain_id,
                                        classname=str(class_name).lower()
                                    )
                            spt_items_dict[id].append(item)

                unlabeled_file = os.path.join(self.split_fewshot_path, f"{file_dname}_unlabeled_p0{self.num_shots}.txt")
                if os.path.exists(unlabeled_file):
                    with open(unlabeled_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_list = line.split()
                            image_path, label = text_list[0], text_list[1]
                            class_name = image_path.split('/')[1]
                            item = Datum(
                                impath=os.path.join(self.dataset_dir, image_path),
                                label=int(label),
                                domain=domain_id,
                                classname=str(class_name).lower()
                            )
                            u_items_dict[id].append(item)

            return spt_items_dict, u_items_dict

        elif input_domains == self.target_domain:
            # dname = input_domains[0]
            if input_domains == "Real World":
                file_dname = "Real"
            else:
                file_dname = input_domains
            domain_id = self.domain_ids[self.target_domain]
            target_items_dict = defaultdict(list)
            target_file = os.path.join(self.split_fewshot_path, f"{file_dname}.txt")
            if os.path.exists(target_file):
                with open(target_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        text_list = line.split()
                        image_path, label = text_list[0], text_list[1]
                        class_name = image_path.split('/')[1]
                        item = Datum(
                            impath=os.path.join(self.dataset_dir, image_path),
                            label=int(label),
                            domain=domain_id,
                            classname=str(class_name).lower()
                        )
                        target_items_dict[0].append(item)

            return target_items_dict

    def creat_batch(self): 

        support_dict = defaultdict(list) 
        support_aug_dict = defaultdict(list)
        for d in range(len(self.source_domain)):
            sample_cls_id_labeled = random.sample(range(len(self.source_support[d])), len(self.source_support[d]) // 2)
            for i in range(len(self.source_support[d])):
                if i in sample_cls_id_labeled:
                    support_dict[d].append(self.source_support[d][i])
                else:
                    support_aug_dict[d].append(self.source_support[d][i])
            # print(len(support_dict[d]))
            # print(len(support_aug_dict[d]))

        support = {} 
        i = 0
        for d in range(len(self.source_domain)):
            support[i] = support_dict[d]
            i += 1

        batch_domain_id = random.sample(range(len(self.source_domain)), len(self.source_domain))  
        batch_domain_aug = [random.randint(0, len(self.source_domain) - 1) for b in batch_domain_id]  

        for id1, id2 in zip(batch_domain_id, batch_domain_aug):
            random.shuffle(support_aug_dict[id1])
            random.shuffle(support_aug_dict[id2])
            support[i] = (support_aug_dict[id1], support_aug_dict[id2])
            i += 1

        self.support = support

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).
        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        label2cname = {label: classname for label, classname in container}
        cname2label = {classname: label for label, classname in container}
        labels = list(label2cname.keys())
        labels.sort()
        classnames = [label2cname[label] for label in labels]
        return label2cname, cname2label, classnames

    def __len__(self):
        if self.mode == "train":
            return len(self.source_domain)*2
        elif self.mode == "test":
            return len(self.test)

    def __getitem__(self, idx):
        if self.mode == "train":
            support_aug = list()
            if idx >= int(len(self.source_domain)*2): #
                return None
            if isinstance(self.support[idx], tuple):
                for it0, it1 in zip(self.support[idx][0], self.support[idx][1]):
                    image_aug1 = Image.open(it0.impath).convert('RGB')
                    image_aug1 = self.transform(image_aug1)

                    image_aug2 = Image.open(it1.impath).convert('RGB')
                    image_aug2 = self.transform(image_aug2)
                    support_aug.append([image_aug1, it0.label, image_aug2, it1.label])
            else:
                for item in self.support[idx]:
                    image = Image.open(item.impath).convert('RGB')
                    image = self.transform(image)
                    support_aug.append([image, item.label])

            return support_aug

        elif self.mode == "test":
            item = self.test[idx]
            image = Image.open(item.impath).convert('RGB')
            image = self.transform(image)
            return image, item.label, item.classname
