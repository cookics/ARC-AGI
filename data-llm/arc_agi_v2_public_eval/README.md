---
license: mit
configs:
- config_name: attempts
  data_files:
  - split: "test"
    path: "**/[0-9a-f]*.json"
  features:
    - name: attempt_1
      dtype: string
    - name: attempt_2
      dtype: string
- config_name: results
  data_files: "**/results.json"
---
