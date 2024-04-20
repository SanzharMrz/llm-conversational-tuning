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

    val_root_dir = root_dir / "val_data_v3"
    if not val_root_dir.exists():
        val_root_dir.mkdir()

    input_data = val_root_dir / "generated_data.json"
    # input_data = val_root_dir / "gd_v2.json"

    # exp_name = "adam"
    # exp_name = "tuned_mistral"
    # exp_name = "llama2_exp6"
    # exp_name = "raw_mistral_7b"
    # exp_name = "raw_llama2"
    # exp_name = "raw_mistral_7b_instruct"

    # model_path = "/home/smurzakhmetov/onceLM/notebooks/llama-v2-base-dataset-no-system-prompt"
    # model_path = "/home/eplotnikov/Once/onceLM/checkpoints/llama2_7b-v1"
    # model_path = "/home/eplotnikov/Once/onceLM/checkpoints/mistral_egor_v1/mistral_forbidden_topics_lora_v4/final_checkpoint") # noqa: E501
    # model_path = "/home/eplotnikov/Once/onceLM/checkpoints/llama2_egor_v4/checkpoint-61064"
    # model_path = "mistralai/Mistral-7B-v0.1"
    # model_path = "TheBloke/Llama-2-7B-fp16"
    # model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    d = [
        ("llama2_sanzhar", "/home/smurzakhmetov/onceLM/notebooks/llama-v2-base-dataset-no-system-prompt"),
        ("adam", "/home/eplotnikov/Once/onceLM/checkpoints/llama2_7b-v1"),
        (
            "tuned_raw_mistral_sanzhar_v1",
            "/home/smurzakhmetov/once/merged_after_lora",  # noqa: E501
        ),
        (
            "tuned_raw_mistral_sanzhar_v2_test",
            "/home/smurzakhmetov/once/merged_after_lorav2",  # noqa: E501
        ),
        ("llama2_sanzhar_moses", "/home/smurzakhmetov/once/moses/llama7b-moses-v1"),
        (
            "tuned_mistral",
            "/home/eplotnikov/Once/onceLM/checkpoints/mistral_egor_v1/mistral_forbidden_topics_lora_v4/final_checkpoint",  # noqa: E501
        ),
        ("llama2_exp6", "/home/eplotnikov/Once/onceLM/checkpoints/llama2_egor_v4/checkpoint-61064"),
        ("raw_mistral_7b", "mistralai/Mistral-7B-v0.1"),
        ("raw_llama2", "TheBloke/Llama-2-7B-fp16"),
        ("raw_mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.2"),
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
            "max_new_tokens": 45,
            "early_stopping": True,
            "num_beams": 7,
            "repetition_penalty": 1.0,
            "remove_invalid_values": True,
            "eos_token_id": 13,
            "pad_token_id": 13,
            "forced_eos_token_id": 13,
            "use_cache": True,
            "no_repeat_ngram_size": 4,
            "num_return_sequences": 1,
        }

        print(generation_config)

        for _ in range(n_turns):
            raw_data = read_data(input_data)
            for i in tqdm.tqdm(range(len(raw_data))):
                sample = raw_data[i]
                if "raw_mistral" not in exp_name:
                    text = f'<s> {sample["profile"]}\n\n{sample["raw"]["raw_placeholders"]["user_name"]}: {sample["question"]}\n{sample["raw"]["raw_placeholders"]["bot_name"]}: '  # noqa: E501
                else:
                    text = f'<s>[INST] {sample["profile"]}[/INST]{sample["raw"]["raw_placeholders"]["user_name"]}: {sample["question"]} </s> {sample["raw"]["raw_placeholders"]["bot_name"]}: '  # noqa: E501
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
