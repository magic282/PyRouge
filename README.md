# PyRouge
Rouge evaluation script implemented with Python.

Currently, only Rouge-N is implemented.

**WARNING** The result is slightly different from the official Rouge.

## Usage

```python
python compute.py ref_file.txt predict_file.txt
```

`ref_file.txt` and `predict_file.txt` are line-by-line text files.