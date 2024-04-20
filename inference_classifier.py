import json

import pandas as pd
import tqdm
from transformers import pipeline


def read_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


MISTAKES_CLASSIFIER_LABELS = [
    "Request for Phone Number",
    "Social Network",
    "Request/Invite to Meet/Date",
    "Normal Response/Question/Sentence",
    "Flirt",
]


def mistakes_score(mistakes_classifier, texts):
    # labels_ok = ["Normal Response/Question/Sentence", "Flirt"]
    res = mistakes_classifier(texts, MISTAKES_CLASSIFIER_LABELS)
    res = [{"text": r["sequence"], **{key: val for key, val in zip(r["labels"], r["scores"])}} for r in res]
    return res
    """
    if res["labels"][0] in labels_ok:
        return "ok", res["scores"][0]
    elif res["labels"][1] in labels_ok and res["scores"][0] < 0.5:
        return "ok", res["scores"][0]
    elif res["labels"][0] not in labels_ok and res["scores"][0] < 0.4:
        return "ok", res["scores"][0]
    else:
        return res["labels"][0], res["scores"][0]
    """


if __name__ == "__main__":
    data_path = "./data/once_dialogues/json_clean/train_instructions.json"
    dataset = read_dataset(data_path)

    device = "cuda:0"
    mistakes_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    results = []
    for sample in tqdm.tqdm(dataset):
        dialogue = sample["dialogue"]
        replies = [r["text"] for r in dialogue if r["speaker"] == "Alice"]
        res = mistakes_score(mistakes_classifier, replies)
        for i in range(len(res)):
            res[i]["session_id"] = sample["session_id"]
        results.extend(res)

    d = pd.DataFrame.from_records(results)
    d.to_csv("./data/once_dialogues/scored_data.csv")
