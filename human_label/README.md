# Generating your own human preferences
Based on the collected indices for queries in this folder, you could also generate your own real human preferences.

## Generating Videos
First, you have to generate videos for queries by running codes below.
```python
python -m JaxPref.human_label_preprocess --env_name {Mujoco env name} --query_path ./human_label  --save_dir {video folder to save} --num_query {number of query} --query_len {query length}
```

## Labeling Human Preferences
After generating videos, code your label and dump it as a pickle with `save_label.py` file.