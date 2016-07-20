import json


def convert_squad(file_path):
	print(file_path)
	question_sets = []
	with open(file_path) as data_file:
		data = json.load(data_file)['data']
		for article in data:
			for paragraph in article['paragraphs']:
				qa_set = {
					'support': parse_support(paragraph),
					'questions': [parse_question(qa_dict) for qa_dict in paragraph['qas']]
				}
				question_sets.append(qa_set)
	return question_sets

def parse_support(para_dict):
	return {
		'src': para_dict['context']
	}

def parse_question(qa_dict):
	# Process answers first...
	# What to do when multiple annotators answer a question??  Majority?
	# Here just the first provided is selected
	chosen_answer = qa_dict['answers'][0]
	chosen_answer_text = chosen_answer['text']
	chosen_answer_start = chosen_answer['answer_start']
	result_answer_dict = {
		'text': chosen_answer_text,
		'span': {
			'start': chosen_answer_start,
			'end': chosen_answer_start + len(chosen_answer_text)
		}
	}
	return {
		'question': qa_dict['question'],
		'answer': result_answer_dict
	}

# Usage:
# From other code, call convert_squad(filename)
# From command line, a single argument converts and writes to stdout
# From command line, two arguments converts arg1 and writes to arg2
def main():
	import sys
	print(sys.argv[1])
	corpus = convert_squad(sys.argv[1])
	if len(sys.argv) == 2:
		print(json.dumps(corpus, indent=2))
	if len(sys.argv) == 3:
		with open(sys.argv[2], 'w') as outfile:
			json.dump(corpus, outfile, indent=2)

if __name__ == "__main__": main()
