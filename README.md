# LaVy: Vietnamese Multimodal Large Language Model

*Pioneering in Vietnamese MLLMs*

## Contents
- [Install](#install)
- [Weights](#llava-weights)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/baochi0212/LaVy
cd ./LaVy
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
python3 -m pip install flash-attn --no-build-isolation
#for xformer: https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers
```

## LaVy Weights
```
#save directory
mkdir ./checkpoints
#download weights
huggingface-cli download Viet-Mistral/Vistral-7B-Chat --local-dir ./checkpoints
huggingface-cli download chitb/LaVy-pretrain --local-dir ./checkpoints
huggingface-cli download chitb/LaVy-instruct --local-dir ./checkpoints
```
### Quickstart Inference 

```
python ./llava/eval/run_llava.py \
   --model_base path_to_Vistral-7B \
   --model_path path_to_LaVy-Instruct \
   --model_name "llava_lora" \
   --conv_mode "mistral_instruct" \
   --image-file "./images/006.jpg" \
   --query "Bạn có biết ai vẽ bức tranh này?"
```
![image](https://github.com/baochi0212/LaVy/assets/77192945/c235c99c-df45-49da-8d11-e1ba3f4441b4)

```
Output:
Bức tranh này được vẽ bởi Leonardo da Vinci, một họa sĩ nổi tiếng người Ý. Bức tranh nổi tiếng này mô tả một người phụ nữ với nụ cười bí ẩn, được gọi là Mona Lisa. Bức tranh được vẽ vào thế kỷ 16 và được coi là một trong những tác phẩm nghệ thuật nổi tiếng và được yêu thích nhất trên thế giới.
```
### Train

1. Prepare
   Set up data & configuration in files in ./scripts, for more information [LlaVA](https://github.com/haotian-liu/LLaVA)

3. Start training
   ```
   #Pretraining
   bash ./training_scripts/pretrain.sh
   #Finetuning
   bash ./training_scripts/finetune.sh


### Evaluation
1. Prepare & Config: 
   Benchmark can be downloaded at: [LaVy](https://huggingface.co/datasets/chitb/LaVy-Bench) [OpenViVQA](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset) 

   Our Gemini Pro prompts for evaluation:
   OpenViVQA:
    ```
    {
    "Question": "sản phẩm của đào tiên được làm tại đâu ?",
    "Label": "sản phẩm của đào tiên được làm tại vietnam",
    "Pred": "Sản phẩm của Đào Tiên được sản xuất tại Việt Nam.",
    "Instruction": "Nhiệm vụ của bạn là với Câu hỏi, Câu trả lời gốc, Câu trả lời dự đoán. Trong đó:\n\nCâu trả lời gốc là câu trả lời đúng cho câu hỏi\nCâu trả lời dự đoán là câu trả lời được dự đoán cho câu hỏi\nNếu câu trả lời dự đoán tương tự câu trả lời gốc và trả lời được câu hỏi thì đó là câu trả lời đúng\nDựa vào đó, hãy kiểm trả xem câu trả lời dự đoán có đúng không. Nếu có trả về 1, nếu không trả về 0. Chỉ trả ra 1 hoặc 0, không giải thích gì thêm.\nCâu hỏi: sản phẩm của đào tiên được làm tại đâu ?\nCâu trả lời gốc: sản phẩm của đào tiên được làm tại vietnam\nCâu trả lời dự đoán: Sản phẩm của Đào Tiên được sản xuất tại Việt Nam.",
    "Response": "1",
    "Status": "success"
    }
    ```
    In the wild
    ```
    {
    "Category": "conv",
    "Question": "Đây là nhân vật lích sử Việt Nam nổi tiếng nào?",
    "Label": "Bức ảnh chụp một người đàn ông đứng trên lễ đài, đang phát biểu trước quần chúng. Ông mặc bộ quần áo kaki, đầu đội mũ ca lô, tay cầm micro. Phía sau ông là một lá cờ đỏ sao vàng. Trên khán đài có rất nhiều người đang ngồi nghe ông phát biểu. Bên dưới khán đài là một đám đông rất đông người, họ đang vẫy cờ và hô vang khẩu hiệu. Bầu trời trong xanh, nắng đẹp. Không khí rất náo nhiệt và phấn khởi.\n\nĐây là Hồ Chí Minh - vị lãnh tụ kính yêu của dân tộc Việt Nam. Người đã dẫn dắt cách mạng Việt Nam giành thắng lợi, thống nhất đất nước, đem lại độc lập, tự do cho dân tộc. Hồ Chí Minh là một người vĩ đại, một anh hùng của dân tộc Việt Nam. Người luôn được nhân dân Việt Nam kính trọng và yêu mến.",
    "Pred": "Nhân vật lịch sử Việt Nam nổi tiếng trong bức ảnh là Hồ Chí Minh, người sáng lập và lãnh đạo Đảng Cộng sản Việt Nam. Ông là một nhà cách mạng, chính trị gia và nhà lãnh đạo quân sự, đóng vai trò quan trọng trong cuộc đấu tranh giành độc lập của Việt Nam khỏi sự cai trị của thực dân Pháp. Bức ảnh cho thấy Hồ Chí Minh đang phát biểu trước đám đông, thể hiện sự ủng hộ và ảnh hưởng của ông đối với người dân Việt Nam.",
    "Instruction": "Bạn là người chấm điểm công bằng. Dưới đây là mô tả chi tiết của một bức ảnh, câu hỏi, và câu trả lời dựa trên bức ảnh đó. Dựa trên tiêu chí:\n- Chính xác: Câu trả lời có thể trả lời câu hỏi không\n- Tự nhiên: Câu trả lời có dùng ngôn ngữ Tiếng Việt tự nhiên không\n- Liên quan: Mục đích câu trả lời là gì và nó có liên quan đến câu hỏi không (câu trả lời có thể lặp từ/ không tự nhiên/ sai sự thật nhưng vẫn liên quan đến câu hỏi.\nSau đo đánh giá câu trả lời dựa trên tháng điểm từ 1 đến 5 (ví dụ 1/5, 2/5). Sau đo giải thích, cho điểm từng tiêu chí và cho ra một số điểm duy nhất trong câu trả lời, và nằm ở cuối câu trả lời:\n\nMô tả: Trong bức ảnh đang có một nhóm bạn gồm 3 nam, 2 nữ. Họ đang làm việc nhóm, có vẻ như là một vấn đề rất căng thẳng.\nCâu hỏi: Mọi người đang làm gì?\nCâu trả lời: Họ đang họp\nĐánh giá: Chính xác: 5/5; Chi tiết: 4/5; Liên quan: 5/5; Điểm cuối cùng: 4.7/5\n\nMô tả: Bức ảnh chụp một người đàn ông đứng trên lễ đài, đang phát biểu trước quần chúng. Ông mặc bộ quần áo kaki, đầu đội mũ ca lô, tay cầm micro. Phía sau ông là một lá cờ đỏ sao vàng. Trên khán đài có rất nhiều người đang ngồi nghe ông phát biểu. Bên dưới khán đài là một đám đông rất đông người, họ đang vẫy cờ và hô vang khẩu hiệu. Bầu trời trong xanh, nắng đẹp. Không khí rất náo nhiệt và phấn khởi.\n\nĐây là Hồ Chí Minh - vị lãnh tụ kính yêu của dân tộc Việt Nam. Người đã dẫn dắt cách mạng Việt Nam giành thắng lợi, thống nhất đất nước, đem lại độc lập, tự do cho dân tộc. Hồ Chí Minh là một người vĩ đại, một anh hùng của dân tộc Việt Nam. Người luôn được nhân dân Việt Nam kính trọng và yêu mến.\nCâu hỏi: Đây là nhân vật lích sử Việt Nam nổi tiếng nào?\nCâu trả lời: Nhân vật lịch sử Việt Nam nổi tiếng trong bức ảnh là Hồ Chí Minh, người sáng lập và lãnh đạo Đảng Cộng sản Việt Nam. Ông là một nhà cách mạng, chính trị gia và nhà lãnh đạo quân sự, đóng vai trò quan trọng trong cuộc đấu tranh giành độc lập của Việt Nam khỏi sự cai trị của thực dân Pháp. Bức ảnh cho thấy Hồ Chí Minh đang phát biểu trước đám đông, thể hiện sự ủng hộ và ảnh hưởng của ông đối với người dân Việt Nam.",
    "Response": "Đánh giá: Chính xác: 5/5; Chi tiết: 5/5; Liên quan: 5/5; Điểm cuối cùng: 5/5",
    "Status": "success"
    }
    ```
3. Batch Inference (Multi-gpu) 
```
#OpenViVQA
bash ./eval_scripts/test_llava.sh $file_name $model_path $model_base $output_file $NUM_GPUs 
#in the wild
bash ./eval_scripts/test_benchmark.sh $file_name $model_path $model_base $output_file $NUM_GPUs
```

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{lavy,
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
