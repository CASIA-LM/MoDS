# MoDS: Model-oriented Data Selection for Instruction Tuning

This repo contains the codes of MoDS to select valuable instruction data from large-scale datasets for a given LLM. 

### Introduction

Instruction tuning has become the de facto method to equip large language models (LLMs) with the ability of following user instructions. Usually, hundreds of thousands or millions of instruction-following pairs are employed to fine-tune the foundation LLMs. Recently, some studies show that a small number of high-quality instruction data is enough. However, how to select appropriate instruction data for a given LLM is still an open problem.  To address this problem, in this paper we present a model-oriented data selection (MoDS) approach, which selects instruction data based on a new criteria considering three aspects: quality, coverage and necessity.  First, our approach utilizes a quality evaluation model to filter out the high-quality subset from the original instruction dataset, and then designs an algorithm to further select from the high-quality subset a seed instruction dataset with good coverage. The seed dataset is applied to fine-tune the foundation LLM to obtain an initial instruction-following LLM. Finally, we develop a necessity evaluation model to find out the instruction data which are performed badly in the initial instruction-following LLM and consider them necessary instructions to further improve the LLMs. In this way, we can get a small high-quality, broad-coverage and high-necessity subset from the original instruction datasets.  As shown in Figure 1, it presents the architecture of our approach.

<div align="center">
  <img src=".\assets\architecture.png" width="80%"/>
</div>


### Environment Dependencies

```shell
transformers==4.31.0
json==2.0.9
pytorch==2.0.1
numpy==1.25.2
sklearn==1.2.1
```

### Stage 1: Quality Evaluation

The quality of instruction data plays a crucial role in the learning of instruction-following capabilities for LLMs. Therefore, to select effective instruction data, we first evaluate the qualities of instruction data and their corresponding response in the large-scale dataset, and then filter out the higher-quality data from it.

- When assessing the qualities of instruction data, we utilize the [reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) model which is developed by OpenAssistant. This is a reward model designed based on the DeBERTa architecture, and is trained on four different types of human feedback data , endowing it with the abilities of QA model evaluation, reward scoring, and detecting potential toxic response via ranking. In this paper, we mainly adopt its reward scoring capability to generate a quality score for each (instruction, input, output) triplet in the large-scale dataset. Consequently, we should download the reward-model-deberta-v3-large-v2 in this step and put it into the folder of "models"

- For the json file from large-scale datasets, we can run the following script to process it and generate a new file with quality scores. "input.json" represents the file from large-scale datasets, while "quality-evaluation.json" represents the output results with quality scores. All files have the same format as [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) 

  ```shell
   cd quality-evaluation
   python  quality-evaluation.py ./input.json ./quality-evaluation.json
  ```

- After computing a quality score for each (instruction, input, output) pair, we will extract the high-quality instruciton data with the following script. "high-quality-data.json" represents the high-quality data we extracted. And "0.0" is the threshold to filter high-quality data. 

  ```shell
  python high-quality-data-extraction.py ./quality-evaluation.json ./high-quality-data.json 0.0
  ```




### Stage 2:  Diverse Data Selection for Seed Instructions

After getting a high-quality instruction dataset, we will further select data from it. In order to select diverse instruction data with the maximum coverage, we propose to use K-Center greedy algorithm for data selection.

- Step 1: In order to compute the sentence embeddings of different instructions, we will download the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model firstly, and then put it into the folder of "models".

- Step 2: After downloading the bert-base-uncased model, we will run the following script to seed instrucitons from high-quality dataset. "top_k" represents the number of seed instructions to be selected.


```shell
cd diverse-data-selection
python run.py ../quality-evaluation/high-quality-data.json ./seed-instructions.json top_k
```

### Stage 3:  Augmented Data Selection

For different LLMs, as the knowledge and capabilities they learned in the pre-training procedure are different, the instruction tuning data they require will be different as well. For one instruction, if the given LLM could generate a good response, it indicates that the given LLM has owned the ability to handle this type of instruction, and this instruction data is not necessary for the fine-tuning of the LLM. Conversely, if the LLM cannot generate a good response, it suggests that the LLM couldn't effectively process that type of instruction data, and the instruction data is very important and unique for the fine-tuning of the target LLM. In this stage, we will extract these instructions with bad responses to build a augmented dataset for the given LLM.

- Step 1: In order to find out these missed instructions, we first fine-tune the pre-trained LLM with the seed instruction dataset, generating an initial LLM. Especially, beforing fine-tuning procedure, we should download the pre-trained [llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) model and put it into the folder of "model".  We can run the following scripts to fine-tune the pre-trained LLM.

  ```
  cd train
  ./run.sh
  ```

  The hyperparameters of  fine-tuning procedure are presented in the following:

  ```
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  torchrun --nproc_per_node=4 --master_port=4568 train.py \
      --model_name_or_path ../models/llama2-7b-hf/ \
      --data_path ../diverse-data-selection/seed-instructions.json \
      --bf16 True \
      --output_dir ../output/initial-model/ \
      --num_train_epochs 3 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 16 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap offload" \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True \
      --gradient_checkpointing True
  ```

- Step 2: After getting the initial LLM, we will use it to inference the responses of all the high-quality instructions. The commands are presented in the following:

  ```
  cd ../inference
  ./instruction_filter.sh
  ```

  The parameters of the instruction_filter.sh file is presented in the following:

  ```
  CUDA_VISIBLE_DEVICES=5 \
  python instruction_filter.py --model_name_or_path ../output/initial-model --instruct_data ../quality-evaluation/high-quality-data.json  --instruct_filtered ../necessity-evaluation/inference.json
  ```

- Step 3: After geting the responses of all the high-quality instrucitons, we will use the necessity evaluation module to compute the review score for each of them, and then extract the instructions with bad responses.  "threshold" represents the value to filter out instructions with bad responses.

  ```
  cd necessity-evaluation
  python  necessity-evaluation.py ./inference.json ./inference-necessity-evaluation.json 
  python necessity-instruction-extraction.py ./inference-necessity-evaluation.json ./necessity-instruction.json threshold
  ```

#### Stage 4: Fine-tuning with Selected Instructions

- Step 1: Merge the seed dataset and augmented dataset.

  ```shell
  cd necessity-evaluation
  python merge.py ../diverse-data-selection/seed-instructions.json ./necessity-instruction.json  ./final-selected-dataset.json
  ```

- Step 2: Fine-tuning the raw LLM with the merged dataset again.

  ```
  cd ../train
  ./run.sh
  ```

  The parameters of training are presented in the following:

  ```
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  torchrun --nproc_per_node=4 --master_port=4568 train.py \
      --model_name_or_path ../models/llama2-7b-hf/ \
      --data_path ../necessity-evaluation/final-selected-dataset.json \
      --bf16 True \
      --output_dir ../output/initial-model/ \
      --num_train_epochs 3 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 16 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap offload" \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True \
      --gradient_checkpointing True
  ```

## Citation
Please cite the paper if you use the data or code in this repo.

```shell
@misc{du2023mods,
      title={MoDS: Model-oriented Data Selection for Instruction Tuning}, 
      author={Qianlong Du and Chengqing Zong and Jiajun Zhang},
      year={2023},
      eprint={2311.15653},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Naturally, you should also cite the work of LLaMA2 and Alpaca.