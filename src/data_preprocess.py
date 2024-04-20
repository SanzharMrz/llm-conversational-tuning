import json
import os
from pathlib import Path
from typing import List, Union

import tqdm


class DialoguesDatasetProcessor:
    def __init__(
        self,
        data_paths: Union[Path, List[Path]],
        max_context_length: int = 20,
        cache_data: bool = False,
        force_preprocess: bool = False,
        dataset_part: str = "train",
        mask_user_reply: bool = False,
    ) -> None:
        assert dataset_part in ["train", "test"]
        if isinstance(data_paths, Path):
            self.data_paths = [
                data_paths,
            ]
        else:
            self.data_paths = data_paths
        self.cache_data = cache_data
        self.force_preprocess = force_preprocess
        self.max_context_length = max_context_length
        self.mask_user_reply = mask_user_reply
        self.data, self.data_raw = [], []

        self.base_cache_dir = Path.cwd() / "data" / "cache"
        self.base_cache_path = self.base_cache_dir / f"{dataset_part}.json"

        if not os.path.exists(self.base_cache_dir):
            os.makedirs(self.base_cache_dir)
        self.is_cached = False
        if os.path.exists(self.base_cache_path):
            self.is_cached = True
        if self.is_cached and not force_preprocess:
            print(f"Load cached data from {self.base_cache_path}")
            self.data = self._read_data(
                [
                    self.base_cache_path,
                ]
            )
        else:
            self.data_raw = self._read_data(self.data_paths)

    def _read_data(self, paths):
        data = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            data.extend(d)
        return data

    def _write_data(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _filter_dialogue(self, dialogue):
        counter = 0
        if len(dialogue) < 2:
            return False, None
        new_dialogue = []
        for reply in dialogue:
            if self.mask_user_reply:
                if reply["speaker"] in ["Alice", "David"]:
                    flag = not (reply["speaker"] == "Alice")
                else:
                    flag = False
            else:
                flag = False
            if "base_filters" in reply["filters"]:
                for k in reply["filters"]["base_filters"]:
                    flag = flag or reply["filters"]["base_filters"][k]
            if "rank_filters" in reply["filters"]:
                if "rules_result" in reply["filters"]["rank_filters"]:
                    flag = flag and (reply["filters"]["rank_filters"]["rules_result"] == "ok")
            if flag:
                counter += 1
            to_train = not flag
            new_dialogue.append(
                {
                    "speaker": reply["speaker"],
                    "text": reply["text"],
                    "to_train": to_train,
                    "order_number": reply["order_number"],
                }
            )
        if not self.mask_user_reply:
            if counter <= len(dialogue) // 2 and new_dialogue[-1]["to_train"]:
                return True, new_dialogue
            else:
                return False, None
        else:
            if counter <= (len(dialogue) // 2 + 6):
                return True, new_dialogue
            else:
                return False, None

    def _split_dialogue(self, dialogue_sample, context_length):
        bio_info = dialogue_sample["bio_info"]
        dialogue = dialogue_sample["dialogue"]
        source = dialogue_sample["source"]
        session_id = dialogue_sample["session_id"]
        topic = "" if "topic" not in dialogue_sample else dialogue_sample["topic"]
        language = "en" if "language" not in dialogue_sample else dialogue_sample["language"]
        dialogue_summary = "" if "dialogue_summary" not in dialogue_sample else dialogue_sample["dialogue_summary"]
        instruction = "" if "instruction" not in dialogue_sample else dialogue_sample["instruction"]

        if not isinstance(topic, str):
            topic = ""

        if language == "fr" and len(instruction) == 0:
            instruction = "Alice should speak in French."

        new_dialogues = []
        filtered_counter = 0
        for i in range(0, len(dialogue), context_length):
            d = dialogue[i : i + context_length]
            filter_result, f_dialogue = self._filter_dialogue(d)
            if filter_result:
                new_dialogues.append(
                    {
                        "bio_info": bio_info,
                        "dialogue": f_dialogue,
                        "source": source,
                        "session_id": session_id,
                        "instruction": instruction,
                        "topic": topic,
                        "language": language,
                        "dialogue_summary": dialogue_summary,
                    }
                )
            else:
                filtered_counter += 1
        return new_dialogues, filtered_counter

    def process(self):
        if self.is_cached and not self.force_preprocess:
            print(f"Current dataset size: {len(self.data)} samples")
            return self.data
        filtered_counter_total = 0
        for dialogue in tqdm.tqdm(self.data_raw):
            samples, filtered_counter = self._split_dialogue(dialogue, self.max_context_length)
            filtered_counter_total += filtered_counter
            self.data.extend(samples)
        print(f"Filtered {filtered_counter_total} samples")
        print(f"Current dataset size: {len(self.data)} samples")
        if self.cache_data:
            print(f"Save cache as {self.base_cache_path}")
            self._write_data(self.data, self.base_cache_path)
        return self.data
