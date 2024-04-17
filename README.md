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
{
"question": "sản phẩm của đào tiên được làm tại đâu ?",
"label": "sản phẩm của đào tiên được làm tại vietnam",
"pred": "Sản phẩm của Đào Tiên được sản xuất tại Việt Nam.",
"instruction": "Nhiệm vụ của bạn là với Câu hỏi, Câu trả lời gốc, Câu trả lời dự đoán. Trong đó:\n\nCâu trả lời gốc là câu trả lời đúng cho câu hỏi\nCâu trả lời dự đoán là câu trả lời được dự đoán cho câu hỏi\nNếu câu trả lời dự đoán tương tự câu trả lời gốc và trả lời được câu hỏi thì đó là câu trả lời đúng\nDựa vào đó, hãy kiểm trả xem câu trả lời dự đoán có đúng không. Nếu có trả về 1, nếu không trả về 0. Chỉ trả ra 1 hoặc 0, không giải thích gì thêm.\nCâu hỏi: sản phẩm của đào tiên được làm tại đâu ?\nCâu trả lời gốc: sản phẩm của đào tiên được làm tại vietnam\nCâu trả lời dự đoán: Sản phẩm của Đào Tiên được sản xuất tại Việt Nam.",
"rewrite_response": "1",
"status": "success"
}
```
In the wild
```
{
"category": "conv",
"question": "Đây là nhân vật lích sử Việt Nam nổi tiếng nào?",
"label": "Bức ảnh chụp một người đàn ông đứng trên lễ đài, đang phát biểu trước quần chúng. Ông mặc bộ quần áo kaki, đầu đội mũ ca lô, tay cầm micro. Phía sau ông là một lá cờ đỏ sao vàng. Trên khán đài có rất nhiều người đang ngồi nghe ông phát biểu. Bên dưới khán đài là một đám đông rất đông người, họ đang vẫy cờ và hô vang khẩu hiệu. Bầu trời trong xanh, nắng đẹp. Không khí rất náo nhiệt và phấn khởi.\n\nĐây là Hồ Chí Minh - vị lãnh tụ kính yêu của dân tộc Việt Nam. Người đã dẫn dắt cách mạng Việt Nam giành thắng lợi, thống nhất đất nước, đem lại độc lập, tự do cho dân tộc. Hồ Chí Minh là một người vĩ đại, một anh hùng của dân tộc Việt Nam. Người luôn được nhân dân Việt Nam kính trọng và yêu mến.",
"pred": "Nhân vật lịch sử Việt Nam nổi tiếng trong bức ảnh là Hồ Chí Minh, người sáng lập và lãnh đạo Đảng Cộng sản Việt Nam. Ông là một nhà cách mạng, chính trị gia và nhà lãnh đạo quân sự, đóng vai trò quan trọng trong cuộc đấu tranh giành độc lập của Việt Nam khỏi sự cai trị của thực dân Pháp. Bức ảnh cho thấy Hồ Chí Minh đang phát biểu trước đám đông, thể hiện sự ủng hộ và ảnh hưởng của ông đối với người dân Việt Nam.",
"instruction": "Bạn là người chấm điểm công bằng. Dưới đây là mô tả chi tiết của một bức ảnh, câu hỏi, và câu trả lời dựa trên bức ảnh đó. Dựa trên tiêu chí:\n- Chính xác: Câu trả lời có thể trả lời câu hỏi không\n- Tự nhiên: Câu trả lời có dùng ngôn ngữ Tiếng Việt tự nhiên không\n- Liên quan: Mục đích câu trả lời là gì và nó có liên quan đến câu hỏi không (câu trả lời có thể lặp từ/ không tự nhiên/ sai sự thật nhưng vẫn liên quan đến câu hỏi.\nSau đo đánh giá câu trả lời dựa trên tháng điểm từ 1 đến 5 (ví dụ 1/5, 2/5). Sau đo giải thích, cho điểm từng tiêu chí và cho ra một số điểm duy nhất trong câu trả lời, và nằm ở cuối câu trả lời:\n\nMô tả: Trong bức ảnh đang có một nhóm bạn gồm 3 nam, 2 nữ. Họ đang làm việc nhóm, có vẻ như là một vấn đề rất căng thẳng.\nCâu hỏi: Mọi người đang làm gì?\nCâu trả lời: Họ đang họp\nĐánh giá: Chính xác: 5/5; Chi tiết: 4/5; Liên quan: 5/5; Điểm cuối cùng: 4.7/5\n\nMô tả: Bức ảnh chụp một người đàn ông đứng trên lễ đài, đang phát biểu trước quần chúng. Ông mặc bộ quần áo kaki, đầu đội mũ ca lô, tay cầm micro. Phía sau ông là một lá cờ đỏ sao vàng. Trên khán đài có rất nhiều người đang ngồi nghe ông phát biểu. Bên dưới khán đài là một đám đông rất đông người, họ đang vẫy cờ và hô vang khẩu hiệu. Bầu trời trong xanh, nắng đẹp. Không khí rất náo nhiệt và phấn khởi.\n\nĐây là Hồ Chí Minh - vị lãnh tụ kính yêu của dân tộc Việt Nam. Người đã dẫn dắt cách mạng Việt Nam giành thắng lợi, thống nhất đất nước, đem lại độc lập, tự do cho dân tộc. Hồ Chí Minh là một người vĩ đại, một anh hùng của dân tộc Việt Nam. Người luôn được nhân dân Việt Nam kính trọng và yêu mến.\nCâu hỏi: Đây là nhân vật lích sử Việt Nam nổi tiếng nào?\nCâu trả lời: Nhân vật lịch sử Việt Nam nổi tiếng trong bức ảnh là Hồ Chí Minh, người sáng lập và lãnh đạo Đảng Cộng sản Việt Nam. Ông là một nhà cách mạng, chính trị gia và nhà lãnh đạo quân sự, đóng vai trò quan trọng trong cuộc đấu tranh giành độc lập của Việt Nam khỏi sự cai trị của thực dân Pháp. Bức ảnh cho thấy Hồ Chí Minh đang phát biểu trước đám đông, thể hiện sự ủng hộ và ảnh hưởng của ông đối với người dân Việt Nam.",
"rewrite_response": "Đánh giá: Chính xác: 5/5; Chi tiết: 5/5; Liên quan: 5/5; Điểm cuối cùng: 5/5",
"status": "success"
}
```

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{lavy,
      title={LaVy: Vietnamese Multimodal Large Language Model}, 
      author={Chi Tran and Huong Le Thanh},
      publisher={arXiv preprint arXiv:2404.07922},
      year={2024}
}
```

## Acknowledgement
We are grateful for:
- [LlaVA](https://github.com/haotian-liu/LLaVA) for LlaVA codebase
- [Viet-Mistral](https://huggingface.co/Viet-Mistral) for Vistral 7B

# LaVy
