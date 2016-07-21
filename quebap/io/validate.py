import json
import jsonschema
from sys import argv

schema = open(argv[1]).read() 
data = open(argv[2]).read()

try:
	jdata = json.loads(data)
	jschema = json.loads(schema)
	jsonschema.validate(jdata, jschema)
except jsonschema.ValidationError as e:
	print(e.message)
except jsonschema.SchemaError as e:
	print(e)

print('JSON successfully validated.')