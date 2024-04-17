import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import jsonlines

#Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)

class LlavaDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, image_processor, model_config):
        self.model_config = model_config
        self.jsonl_path = jsonl_path
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        with jsonlines.open(self.jsonl_path) as reader:
            return len(list(reader))  # Count lines

    def __getitem__(self, index):
        with jsonlines.open(self.jsonl_path) as reader:
            for i, line in enumerate(reader):
                if i == index:
                    image_path = "./benchmark_vi/images_official/" + line['image_id']
                    question = line['question']
                    answer = line['answer']
                    qs = line['question']
                    cat = line['category']
                    if self.model_config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    print("Prompt", prompt)

                    # Load image (replace with your image loading logic)
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = process_images([image], self.image_processor, self.model_config)[0]

                    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

                    return input_ids, image_tensor, image.size, self.tokenizer(question, return_tensors='pt').input_ids, self.tokenizer(answer, return_tensors='pt').input_ids, self.tokenizer(cat, return_tensors='pt').input_ids


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, question, answer, cat = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    question = torch.stack(question, dim=0)
    answer = torch.stack(answer, dim=0)
    cat = torch.stack(cat, dim=0)
    return input_ids, image_tensors, image_sizes, question, answer, cat


# DataLoader
def create_data_loader(jsonl_path, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = LlavaDataset(jsonl_path, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'llava_lora' if args.pretrain == 'false' else 'llava'
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    writer = jsonlines.open(f"output_{os.path.basename(args.file)}", "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(args.file,  tokenizer, image_processor, model.config)

    for input_ids, image_tensor, image_sizes, question, answer, cat in tqdm(data_loader):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        #print("????", input_ids.shape, question.shape, answer.shape)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=50,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        question = tokenizer.batch_decode(question.squeeze(0), skip_special_tokens=True)[0].strip()
        answer = tokenizer.batch_decode(answer.squeeze(0), skip_special_tokens=True)[0].strip()
        cat = tokenizer.batch_decode(cat.squeeze(0), skip_special_tokens=True)[0].strip()
        #ans_id = shortuuid.uuid()
        writer.write({"category": cat, "question": question, "label": answer, "pred": outputs})
#        # ans_file.flush()
#    #ans_file.close()
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="vistral")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--pretrain", type=str)
    args = parser.parse_args()

    eval_model(args)
