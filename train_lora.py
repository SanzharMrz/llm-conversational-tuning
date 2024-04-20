import json
import os
import random
from functools import partial
from pathlib import Path

import torch
from bitsandbytes.optim import Adam
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from src.data_preprocess import DialoguesDatasetProcessor
from src.dialogue_dataset import DialogueDataset, collate_fn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


os.environ["WANDB_PROJECT"] = "llm_train"

CONTEXT_LEN = 1024

DEVICE = torch.device("cuda")

random.seed(42)

if __name__ == "__main__":
    # model_name = "TheBloke/Llama-2-7B-fp16"
    # model_name = "HuggingFaceH4/zephyr-7b-beta"

    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "./checkpoints/mistral_egor_v1/checkpoint-61720"

    exp_name = "mistral_forbidden_topics_lora_v6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # with open("./data/once_forbidden_topics/json_clean/train_mask_user.json", "r", encoding="utf-8") as f:
    #    train = json.load(f)

    train = DialoguesDatasetProcessor(
        [
            # Path("./data/once_dialogues_fr/json_clean/once_dialogues_translated.json"),
            # Path("./data/once_dialogues_fr/json_clean/train_once_main_fem_new_bio.json"),
            # Path("./data/once_dialogues_fr/json_clean/train_forbidden_topics_new_bio_v2.json"),
            # Path("./data/once_dialogues_fr/json_clean/train_once_main_mal.json"),
            Path("./data/once_forbidden_topics/json_clean/train_no_topics_instructions_changed_bio.json"),
            Path(
                "/home/smurzakhmetov/once/datasaet_curation/data_from_google_drive/[once]sia_starters_remastered.json"
            ),
            Path("/home/smurzakhmetov/once/datasaet_curation/data_from_google_drive/[once]sia_remastered.json"),
            # Path(
            #    "/home/smurzakhmetov/once/datasaet_curation/data_from_google_drive/[once]sia_forbiddens_remastered.json"
            # ),
        ],
        max_context_length=21,
        cache_data=False,
        force_preprocess=True,
        dataset_part="train",
        mask_user_reply=True,
    ).process()

    # train.extend(train_fr)

    # random.shuffle(train)

    test = DialoguesDatasetProcessor(
        Path("./data/eva_ai_cleaned/json_clean/test.json"),
        max_context_length=20,
        cache_data=False,
        force_preprocess=True,
        dataset_part="test",
    ).process()[:100]

    train_dataset = DialogueDataset(
        train,
        tokenizer,
        max_length=CONTEXT_LEN,
        mode="train",
    )

    eval_dataset = DialogueDataset(
        test,
        tokenizer,
        max_length=CONTEXT_LEN,
        mode="test",
    )

    print(f"Train: {len(train_dataset)}, test: {len(eval_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(DEVICE)

    args = {
        "epochs": 1,
        "bs": 3,
        "warmup": 0.5,
        "grad_accum": 2,
        "lr": 5 * 1e-6,
        "tokens": CONTEXT_LEN,
        "training": False,
    }

    # train_dataset = dataset  # TextDataset(tokenizer=tokenizer,file_path=TRAIN_TXT, block_size=args["tokens"])
    data_collator = partial(
        collate_fn, tokenizer=tokenizer
    )  # DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model.training = args["training"]
    # model.to(DEVICE);

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    path = f"/home/eplotnikov/Once/onceLM/checkpoints/mistral_egor_v1/{exp_name}"

    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        do_predict=True,
        gradient_checkpointing=True,
        output_dir=path,
        overwrite_output_dir=True,
        num_train_epochs=args["epochs"],
        per_device_train_batch_size=args["bs"],
        per_device_eval_batch_size=args["bs"],
        warmup_ratio=args["warmup"],
        gradient_accumulation_steps=args["grad_accum"],
        learning_rate=args["lr"],
        # use_cache = False,
        save_strategy="steps",
        save_steps=0.2,
        evaluation_strategy="steps",
        eval_steps=0.2,
        report_to=["wandb", "tensorboard"],
        run_name=exp_name,
        save_total_limit=3,
        fp16=False,
    )

    num_training_steps = len(train_dataset)
    num_warmup_steps = 100

    optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=0, optim_bits=8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
    )

    output = trainer.train()
