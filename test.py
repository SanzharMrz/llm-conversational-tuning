import json
from pathlib import Path

import torch
import tqdm
import transformers


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    n_turns = 5

    root_dir = Path("./data")
    if not root_dir.exists():
        root_dir.mkdir()

    val_root_dir = root_dir / "val_data"
    if not val_root_dir.exists():
        val_root_dir.mkdir()

    input_data = val_root_dir / "generated_data.json"

    # exp_name = "adam"
    # exp_name = "tuned_mistral"
    # exp_name = "llama2_exp6"
    # exp_name = "raw_mistral_7b"
    # exp_name = "raw_llama2"
    # exp_name = "raw_mistral_7b_instruct"

    # model_path = "/home/smurzakhmetov/onceLM/notebooks/llama-v2-base-dataset-no-system-prompt"
    # model_path = "/home/eplotnikov/Once/onceLM/checkpoints/llama2_7b-v1"
    # model_path = "/home/eplotnikov/Once/onceLM/checkpoints/mistral_egor_v1/mistral_forbidden_topics_lora_v4/final_checkpoint")
    # model_path = "/home/eplotnikov/Once/onceLM/checkpoints/llama2_egor_v4/checkpoint-61064"
    # model_path = "mistralai/Mistral-7B-v0.1"
    # model_path = "TheBloke/Llama-2-7B-fp16"
    # model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    d = [
        ("llama2_sanzhar", "/home/smurzakhmetov/onceLM/notebooks/llama-v2-base-dataset-no-system-prompt"),
        ("adam", "/home/eplotnikov/Once/onceLM/checkpoints/llama2_7b-v1"),
        (
            "tuned_mistral",
            "/home/eplotnikov/Once/onceLM/checkpoints/mistral_egor_v1/mistral_forbidden_topics_lora_v4/final_checkpoint",
        ),
        ("llama2_exp6", "/home/eplotnikov/Once/onceLM/checkpoints/llama2_egor_v4/checkpoint-61064"),
        ("raw_mistral_7b", "mistralai/Mistral-7B-v0.1"),
    ]

    for exp_name, model_path in d:
        print(f"Current exp: {exp_name}")
        exp_root_dir = val_root_dir / exp_name
        if not exp_root_dir.exists():
            exp_root_dir.mkdir()

        device = torch.device("cuda:0")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()

        generation_config = {
            "do_sample": True,
            "temperature": 1.3,
            "top_p": 0.85,
            "early_stopping": True,
            "max_time": 2.0,
            "min_new_tokens": 5,
            "max_new_tokens": 40,
            "num_beams": 2,
            "eos_token_id": 13,
            "pad_token_id": 13,
            "forced_eos_token_id": 13,
            "use_cache": True,
            "remove_invalid_values": True,
            "no_repeat_ngram_size": 6,
            "repetition_penalty": 1.5,
            "length_penalty": 2.0,
            "num_return_sequences": 2,
        }

        for _ in range(n_turns):
            raw_data = read_data(input_data)
            for i in tqdm.tqdm(range(len(raw_data))):
                sample = raw_data[i]
                text = f'<s> {sample["profile"]}\n\n{sample["raw"]["raw_placeholders"]["user_name"]}: {sample["question"]}\n{sample["raw"]["raw_placeholders"]["bot_name"]}: '
                # text = f'<s>[INST] {sample["profile"]} [/INST]</s><s>[INST] {sample["raw"]["raw_placeholders"]["user_name"]}: {sample["question"]} [/INST]</s><s>[INST] {sample["raw"]["raw_placeholders"]["bot_name"]}: '
                model_input = tokenizer(text, return_tensors="pt")
                context_len = model_input["attention_mask"].sum().item()
                model_input = model_input.to(device)
                with torch.inference_mode():
                    generated_ids = model.generate(**model_input, **generation_config).detach().cpu()
                response_ids = generated_ids[:, context_len:]
                response_text = tokenizer.decode(response_ids[0])
                response_text = response_text.replace("\n", "").strip()
                raw_data[i]["responses"] = {f"{exp_name}_response": response_text}
            t_name = f"turn_{_ + 1}.json"
            print(f"Write to {exp_root_dir / t_name}")
            write_data(raw_data, exp_root_dir / t_name)

        del model
        torch.cuda.empty_cache()
