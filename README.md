# Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning

Install instructions:

We use python3.10, cuda12.3 on an NVIDIA A100-80GB for all experiments. Please install the following python packages:

``` bash
pip install torch numpy huggingface ninja termcolor tqdm transformers accelerate tyro wandb datasets peft schedulefree
```

To reproduce our models, cd into codebase and run scripts/script_trainall.sh.
