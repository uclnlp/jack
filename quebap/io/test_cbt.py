import validate

def test_cbt_snippet_format():
    response = validate.main('../data/snippet/CBT/cbt_NE_train_snippet.quebap.json', 'dataset_schema.json')
    assert response == 'JSON successfully validated.'
