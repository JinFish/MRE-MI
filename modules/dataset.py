import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ViTImageProcessor


class MREProcessor(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.re_path = os.path.join(self.data_path, "ours_rel2id.json")
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, use_fast=False)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<h>', '</h>', '<t>', '</t>']})


    def load_from_file(self, file_name):
        load_file = os.path.join(self.data_path, file_name)
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, images  = [], [], [], [], [] 
            for i, line in enumerate(lines):
                line = json.loads(line)
                words.append(line['tokens'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                images.append(line['images'])

        assert len(words) == len(relations) == len(heads) == len(tails) == len(images)

        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'images': images}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding='utf-8') as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

class MNRE_MI_Dataset(Dataset):
    def __init__(self, processor:MREProcessor, args, file_name) -> None:
        self.processor:MREProcessor = processor
        self.args = args
        self.image_processor:ViTImageProcessor = ViTImageProcessor.from_pretrained(args.image_model_name)
        self.data_path = args.data_path
        self.image_path = args.image_path
        self.aux_image_path = args.aux_image_path
        self.data_dict = self.processor.load_from_file(file_name)
        self.re_dict = self.processor.get_relation_dict()
        self.aux_image_dict = torch.load(args.aux_image_dict)
        self.device = args.device
        self.tokenizer = self.processor.tokenizer
        self.max_seq_length = args.max_seq_length

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, images = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                   self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                   self.data_dict['images'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<h>')
            if i == head_pos[1]:
                extend_word_list.append('</h>')
            if i == tail_pos[0]:
                extend_word_list.append('<t>')
            if i == tail_pos[1]:
                extend_word_list.append('</t>')
            extend_word_list.append(word_list[i])
        if "</t>" not in extend_word_list:
            extend_word_list.append("</t>")
        if "</h>" not in extend_word_list:
            extend_word_list.append("</h>")
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq_length, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict["token_type_ids"], encode_dict['attention_mask']

        if "L" in self.args.image_model_name:
            if "vg" in self.args.aux_image_path:
                main_images = torch.zeros((4,3,384,384))
                aux_images = torch.zeros((8,3,384,384))
            else:
                main_images = torch.zeros((4,3,384,384))
                aux_images = torch.zeros((12,3,384,384))
        else:
            if "vg" in self.args.aux_image_path:
                main_images = torch.zeros((4,3,224,224))
                aux_images = torch.zeros((8,3,224,224))
            else:
                main_images = torch.zeros((4,3,224,224))
                aux_images = torch.zeros((12,3,224,224))
            
        for i, image in enumerate(images):
            img = None
            try:
                img = Image.open(os.path.join(self.image_path, image)).convert("RGB")
            except:
                img = Image.open(os.path.join(self.image_path, "inf.jpg"))
            pixel_values = self.image_processor(img, return_tensors='pt')["pixel_values"][0]
            main_images[i] = pixel_values
            aux_images_names = self.aux_image_dict[image]
            for j, aux_images_name in enumerate(aux_images_names):
                aux_img = Image.open(os.path.join(self.aux_image_path, aux_images_name)).convert("RGB")
                pixel_values = self.image_processor(aux_img, return_tensors='pt')["pixel_values"][0]
                aux_images[i*2+j] = pixel_values


        re_label = self.re_dict[relation]  # label to id

        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
            main_images, aux_images, torch.tensor(re_label)