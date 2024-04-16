# LaVy: Vietnamese Multimodal Large Language Model

*Pioneering in Vietnamese MLLMs*

## Contents
- [Install](#install)
- [Weights](#llava-weights)
- [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/baochi0212/LaVy
cd LLaVA
```

2. Install Package
```Shell
conda create --prefix llava_env python=3.11
conda activate ./llava_env
python3 -m pip install -e .
```

3. Install additional packages for training cases
```
#flash attention
pip install flash-attn --no-build-isolation
#for xformer: https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers
```


### Quick Start With HuggingFace

<details>
<summary>Example Code</summary>

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py`.

You can also use the `eval_model` function in `llava/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

``` python
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```
</details>

## LaVy Weights
```
#save directory
mkdir checkpoints
#download weights
huggingface-cli download Viet-Mistral/Vistral-7B-Chat --local-dir ./checkpoints
huggingface-cli download chitb/LaVy-pretrain --local-dir ./checkpoints
huggingface-cli download chitb/LaVy-instruct --local-dir ./checkpoints
```


### Train

1. Prepare data & config
   Set up data & configuration in files in ./scripts 

3. Start training
   ```
   #Pretraining
   bash ./scripts/pretrain.sh
   #Finetuning
   bash ./scripts/finetune.sh


### Evaluation
LaVy Benchmark can be downloaded at: 
Our Gemini Pro prompts for evaluation:
OpenViVQA:
```
{"question": "sản phẩm của đào tiên được làm tại đâu ?", "label": "sản phẩm của đào tiên được làm tại vietnam", "pred": "Sản phẩm của Đào Tiên được sản xuất tại Việt Nam.", "instruction": "Nhiệm vụ của bạn là với Câu hỏi, Câu trả lời gốc, Câu trả lời dự đoán. Trong đó:\n\n*Câu trả lời gốc là câu trả lời đúng cho câu hỏi\n*Câu trả lời dự đoán là câu trả lời được dự đoán cho câu hỏi\n*Nếu câu trả lời dự đoán tương tự câu trả lời gốc và trả lời được câu hỏi thì đó là câu trả lời đúng\nDựa vào đó, hãy kiểm trả xem câu trả lời dự đoán có đúng không. Nếu có trả về 1, nếu không trả về 0. Chỉ trả ra 1 hoặc 0, không giải thích gì thêm.\n*Câu hỏi: sản phẩm của đào tiên được làm tại đâu ?\n*Câu trả lời gốc: sản phẩm của đào tiên được làm tại vietnam\n*Câu trả lời dự đoán: Sản phẩm của Đào Tiên được sản xuất tại Việt Nam.", "rewrite_response": "1", "status": "success"}
```
In the wild
```

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```

## Acknowledgement
We are grateful for:
- [LlaVA](https://github.com/haotian-liu/LLaVA) for LlaVA codebase
- [Viet-Mistral](https://huggingface.co/Viet-Mistral) for Vistral 7B

# LaVy
