- Mapping to Quebap format
`$ python3 quebap/io/SNLI2quebap_v1.py`
- Validating format
`$ python3 quebap/io/validate.py ./quebap/data/snippet/SNLI_v1/snippet_quebapformat.json quebap/io/dataset_schema.json`
- Debugging
`$ python3 quebap/model/reader.py --train quebap/data/SNLI/snli_1.0/snli_1.0_debug_quebap.jsonl --test quebap/data/SNLI/snli_1.0/snli_1.0_debug_quebap.jsonl`