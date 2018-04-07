import json
import sys


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def convert2SQUAD_format(hoppy_data, write_file_name):
    """
    Converts QAngaroo data (hoppy_data) into SQuAD format.
    The SQuAD-formatted data is written to disk at write_file_name.
    Note: All given support documents per example are concatenated
        into one super-document. All text is lowercased.
    """
    # adapt the JSON tree structure used in SQUAD.
    squad_formatted_content = dict()
    squad_formatted_content['version'] = 'hoppy_squad_format'
    data = []

    # loop over dataset
    for datum in hoppy_data:

        # Format is deeply nested JSON -- prepare data structures
        data_ELEMENT = dict()
        data_ELEMENT['title'] = 'dummyTitle'
        paragraphs = []
        paragraphs_ELEMENT = dict()
        qas = []
        qas_ELEMENT = dict()
        qas_ELEMENT_ANSWERS = []
        ANSWERS_ELEMENT = dict()


        ### content start
        qas_ELEMENT['id'] = datum['id']
        qas_ELEMENT['question'] = datum['query']

        # concatenate all support documents into one superdocument
        superdocument = " <new_doc> ".join(datum['supports']).lower()

        # where is the answer in the superdocument?
        answer_position = superdocument.find(datum['answer'].lower())
        if answer_position == -1:
            continue

        ANSWERS_ELEMENT['answer_start'] = answer_position
        ANSWERS_ELEMENT['text'] = datum['answer'].lower()
        ### content end


        # recursively fill in content into the nested SQuAD data format
        paragraphs_ELEMENT['context'] = superdocument
        qas_ELEMENT_ANSWERS.append(ANSWERS_ELEMENT)

        qas_ELEMENT['answers'] = qas_ELEMENT_ANSWERS
        qas.append(qas_ELEMENT)

        paragraphs_ELEMENT['qas'] = qas
        paragraphs.append(paragraphs_ELEMENT)

        data_ELEMENT['paragraphs'] = paragraphs
        data.append(data_ELEMENT)

    squad_formatted_content['data'] = data

    with open(write_file_name, 'w') as f:
        json.dump(squad_formatted_content, f, indent=1)

    print('Done writing SQuAD-formatted data to: ',write_file_name)




def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert2SQUAD_format(load_json(input_path), output_path)


if __name__ == "__main__":
    main()
