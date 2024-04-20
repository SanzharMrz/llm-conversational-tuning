import os
from functools import partial
from pathlib import Path

import torch
from bitsandbytes.optim import Adam
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

if __name__ == "__main__":
    # model_name = "TheBloke/Llama-2-7B-fp16"
    # model_name = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(DEVICE)

    train = DialoguesDatasetProcessor(
        [
            # Path("./data/eva_ai_cleaned/json_clean/train.json"),
            Path("./data/once_dialogues/json_clean/train_instructions.json"),
            # Path("./data/once_dialogues_fr/json_clean/train_forbidden_topics.json"),
            # Path("./data/once_dialogues_fr/json_clean/train_once_main_fem.json"),
            # Path("./data/once_dialogues_fr/json_clean/train_once_main_mal.json"),
        ],
        max_context_length=19,
        cache_data=False,
        force_preprocess=True,
        dataset_part="train",
        mask_user_reply=True,
    ).process()
    test = DialoguesDatasetProcessor(
        Path("./data/eva_ai_cleaned/json_clean/test.json"),
        max_context_length=19,
        cache_data=False,
        force_preprocess=True,
        dataset_part="test",
    ).process()

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

    args = {
        "epochs": 1,
        "bs": 1,
        "warmup": 0.5,
        "grad_accum": 1,
        "lr": 2 * 1e-7,
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

    path = "/home/eplotnikov/Once/onceLM/checkpoints/mistral_egor_v1"

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
        report_to=[
            "wandb",
            "tensorboard",
        ],
        run_name="mistral_egor",
        save_total_limit=3,
        fp16=False,
    )

    num_training_steps = len(train_dataset)
    num_warmup_steps = 5000

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
