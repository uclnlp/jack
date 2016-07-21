import json
import jsonschema
from sys import argv

schema = eval(open(argv[1]).read())
data = eval(open(argv[2]).read())

try:
	jdata = json.loads(data)
	jschema = json.loads(schema)
	jsonschema.validate(jdata, jschema)
	print('JSON successfully validated.')
except jsonschema.ValidationError as e:
	print(e.message)
except jsonschema.SchemaError as e:
	print(e)
