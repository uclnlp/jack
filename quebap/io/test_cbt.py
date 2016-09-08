import quebap.io.validate as validate

def test_cbt_snippet_format():
    response = validate.main('quebap/data/snippet/CBT/cbt_NE_train_snippet.quebap.json', 'quebap/io/dataset_schema.json')
    assert response == 'JSON successfully validated.'


test_cbt_snippet_format()
