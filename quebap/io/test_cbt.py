import quebap.io.validate as validate

def test_cbt_snippet_format():
    response = validate.main('quebap/data/CBT/snippet_quebapformat.json', 'quebap/io/dataset_schema.json')
    assert response == 'JSON successfully validated.'


test_cbt_snippet_format()
