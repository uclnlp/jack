import json
import jsonschema
from sys import argv

with open(argv[1]) as f:
    data = json.load(f)

with open(argv[2]) as f:
    schema = json.load(f)



try:
	jsonschema.validate(data, schema)
	print('JSON successfully validated.')
except jsonschema.ValidationError as e:
	print(e.message)
except jsonschema.SchemaError as e:
	print(e)
