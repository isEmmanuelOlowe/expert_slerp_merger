# MOE Merger
Merge moe layers and make a smaller model


install the script by:
```bash
git clone https://github.com/isEmmanuelOlowe/moe_merger
cd moe_merger
pip install -r requirements.txt
```

This will convert Jamba 16 experts into 4 experts.

The notebook is available here [here.](https://github.com/isEmmanuelOlowe/moe_merger/blob/master/jamba_merger.ipynb)

```bash
python main.py --model_id "ai21labs/Jamba-v0.1" --temp_dir "tmp/temp" --save_dir "tmp/Jamba-4xMoE_slerp" --num_experts_per_tok=2 --num_local_experts=4
```


This script is inspired by work on Mistral Merger availabele [here.](https://huggingface.co/mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1/blob/main/notebook/convert_mixtral_8x7b_to_4x7b.ipynb)