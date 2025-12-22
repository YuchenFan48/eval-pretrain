---
license: apache-2.0
dataset_info:
  features:
  - name: task_id
    dtype: int64
  - name: code
    dtype: string
  - name: prompt
    dtype: string
  - name: source_file
    dtype: string
  - name: test_imports
    sequence: string
  - name: test_list
    sequence: string
  - name: test
    dtype: string
  splits:
  - name: test
    num_bytes: 4841266
    num_examples: 378
  download_size: 1129135
  dataset_size: 4841266
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
