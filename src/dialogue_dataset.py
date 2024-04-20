from typing import Any, Dict, List, Union

import torch
import transformers
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    def __init__(
        self,
        dataset: List[Any],
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        max_length: int = 1024,
        mode: str = "train",
        model_type: str = "llama",
    ) -> None:
        """
        Data format:
        <s> bot info <instruction> instruction to bot (can be empty) \n\n
        bot_name: bot_reply_1\nuser_name: user_reply_1\n...
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.model_type = model_type
        assert mode in ["train", "test"]
        assert model_type in ["llama", "zephyr"]

    def __len__(self) -> int:
        return len(self.dataset)

    def _preprocess_dialogue_llama(self, sample):
        """
        formatting: "<s> {bio_info} <instruction> {instruction} \n\n" + \
        ' ' + '\n '.join([f'{s["speaker"]}: {s["text"]}' for s in dialogue[-10:]]) + '\n'
        """

        bio_info = sample["bio_info"]
        dialogue = sample["dialogue"]
        instruction = sample["instruction"] if "instruction" in sample else ""
        # dialogue_summary = sample["dialogue_summary"] if "dialogue_summary" in sample else ""
        language = sample["language"]
        topic = sample["topic"]

        base_prompt = f"<s> {bio_info}"
        if len(instruction) > 0:
            base_prompt = f"{base_prompt} <instruction> {instruction} </instruction>"
        if len(language) > 0:
            base_prompt = f"{base_prompt} <language> {language} </language>"
        if len(topic) > 0:
            base_prompt = f"{base_prompt} <topic> {topic} </topic>"
        base_prompt = f"{base_prompt} \n\n"

        tokenized_base_prompt = self.tokenizer.encode(base_prompt, add_special_tokens=False)
        input_ids, labels, attention_mask = (tokenized_base_prompt[:], tokenized_base_prompt[:], [])
        tokenized_dialogue = []
        for reply in dialogue:
            str_reply = f'{reply["speaker"]}: {reply["text"]}\n'
            tokenized_reply = self.tokenizer.encode(str_reply, add_special_tokens=False)
            tokenized_dialogue.append(
                {
                    "input_ids": tokenized_reply,
                    "labels": tokenized_reply
                    if reply["to_train"] or self.mode == "test"
                    else [
                        -100,
                    ]
                    * len(tokenized_reply),
                }
            )
        total_len = sum([len(t["input_ids"]) for t in tokenized_dialogue]) + len(tokenized_base_prompt)
        while total_len > self.max_length:
            tokenized_dialogue = tokenized_dialogue[1:]
            total_len = sum([len(t["input_ids"]) for t in tokenized_dialogue]) + len(tokenized_base_prompt)
        for t_d in tokenized_dialogue:
            input_ids.extend(t_d["input_ids"])
            labels.extend(t_d["labels"])
        assert len(input_ids) == len(labels) and len(input_ids) <= self.max_length
        attention_mask = [
            1,
        ] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def _preprocess_dialogue_zephyr(self, sample):
        bio_info = sample["bio_info"]
        dialogue = sample["dialogue"]
        instruction = sample["instruction"] if "instruction" in sample else ""
        # dialogue_summary = sample["dialogue_summary"] if "dialogue_summary" in sample else ""
        language = sample["language"]
        topic = sample["topic"]

        base_prompt = "<s><|system|>\nUser send you dialogue between two persons."
        base_prompt = f"{base_prompt} You must write answer based on information below."
        base_prompt = f"{base_prompt} Description for one of the persons: {bio_info}"
        if len(instruction) > 0:
            base_prompt = f"{base_prompt} Instruction to follow for person with description: {instruction}"
        if len(language) > 0:
            base_prompt = f"{base_prompt} You should write answer using {language} language."
        if len(topic) > 0:
            base_prompt = f"{base_prompt} Your answer should follow {topic} topic."

        base_prompt = f"{base_prompt}</s>\n<|user|>\nDialogue:\n"

        tokenized_base_prompt = self.tokenizer.encode(base_prompt, add_special_tokens=False)
        input_ids, labels, attention_mask = (tokenized_base_prompt[:], tokenized_base_prompt[:], [])
        tokenized_dialogue = []
        for reply in dialogue[:-1]:
            str_reply = f'{reply["speaker"]}: {reply["text"]}\n'
            tokenized_reply = self.tokenizer.encode(str_reply, add_special_tokens=False)
            tokenized_dialogue.append(
                {
                    "input_ids": tokenized_reply,
                    "labels": tokenized_reply,
                }
            )

        assistant_dialogue = "</s>\n<|assistant|>\n"
        assistant_dialogue = f"{assistant_dialogue}{dialogue[-1]['speaker']}: {dialogue[-1]['text']}</s>"
        tokenized_assistant_dialogue = self.tokenizer.encode(assistant_dialogue, add_special_tokens=False)

        total_len = (
            sum([len(t["input_ids"]) for t in tokenized_dialogue])
            + len(tokenized_base_prompt)
            + len(tokenized_assistant_dialogue)
        )
        while total_len > self.max_length:
            tokenized_dialogue = tokenized_dialogue[1:]
            total_len = (
                sum([len(t["input_ids"]) for t in tokenized_dialogue])
                + len(tokenized_base_prompt)
                + len(tokenized_assistant_dialogue)
            )
        for t_d in tokenized_dialogue:
            input_ids.extend(t_d["input_ids"])
            labels.extend(t_d["labels"])

        input_ids.extend(tokenized_assistant_dialogue)
        labels.extend(tokenized_assistant_dialogue)

        attention_mask = [
            1,
        ] * len(input_ids)

        assert len(input_ids) == len(labels) and len(input_ids) <= self.max_length

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self.dataset[index]
        if self.model_type == "llama":
            preprocessed_sample = self._preprocess_dialogue_llama(sample)
        else:
            preprocessed_sample = self._preprocess_dialogue_zephyr(sample)
        return preprocessed_sample


def collate_fn(batch, tokenizer, return_tensors=None):
    merged_sample = {}
    for sample in batch:
        for key in sample:
            if key in merged_sample:
                merged_sample[key].append(sample[key])
            else:
                merged_sample[key] = [
                    sample[key],
                ]
    padded_sample = tokenizer.pad(merged_sample, padding="longest")  # except labels
    for i in range(len(padded_sample["input_ids"])):
        add_len = len(padded_sample["input_ids"][i]) - len(padded_sample["labels"][i])
        padded_sample["labels"][i] = (
            [
                -100,
            ]
            * add_len
        ) + padded_sample[
            "labels"
        ][i]
        assert (
            len(padded_sample["input_ids"][i])
            == len(padded_sample["labels"][i])
            == len(padded_sample["attention_mask"][i])
        )
    padded_sample["labels"] = torch.LongTensor(padded_sample["labels"])
    padded_sample["input_ids"] = torch.LongTensor(padded_sample["input_ids"])
    padded_sample["attention_mask"] = torch.LongTensor(padded_sample["attention_mask"])
    return padded_sample
