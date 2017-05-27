#!/usr/bin/env python3

import json
import jsonschema
from sys import argv

def main(arg1, arg2):
    with open(arg1) as f:
        data = json.load(f)

    with open(arg2) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(data, schema)
        return 'JSON successfully validated.'
    except jsonschema.ValidationError as e:
        return e.message
    except jsonschema.SchemaError as e:
        return e


if __name__ == '__main__':
    response = main(argv[1], argv[2])
    print(response)
