- Mapping to jtr format
`$ python3 jtr/io/SNLI2jtr_v1.py`
- Validating format
`$ python3 jtr/io/validate.py ./jtr/data/snippet/SNLI_v1/snippet_jtrformat.json jtr/io/dataset_schema.json`
- Debugging
`$ python3 jtr/model/reader.py --train jtr/data/SNLI/snli_1.0/snli_1.0_debug_jtr.jsonl --test jtr/data/SNLI/snli_1.0/snli_1.0_debug_jtr.jsonl`