# OnceLM - training repo for Once models
---
## Eva AI Stable bot reproducer + new bot with LoRA and fixed prompts

See `notebooks/`

----

## New bot for Once.
---
### Folder structure

* `conf/` - yaml configuration files with params for different models
* `checkpoints/` - intermediate dir to keep current checkpoints. Should be backed up to `~/storage/models/<yourmodel>` when training is done.
* `data/` - processed data for training only. Should be pulled from `~/storage/data/<datasetname>`. Saved and hosted by `dvc` tool. 
* `logs/` - logging of your training.
* `contrib/` - you can put some open-source code here. Currently mostly **Eva AI** original training pipelines are here. Do not commit this directory to Gitlab.
* `notebooks/` - mostly for EDA and examples. Do not put training loops in notebooks.
* `src/` - all you code for training and evaluation. Also contains data processing pipelines.