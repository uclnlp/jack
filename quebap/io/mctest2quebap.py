import json

labels = ["A", "B", "C", "D"]

def clean_mctest_text(text):
    return text.replace('\\newline', '  ')

def convert_mctest(tsv_file, ans_file):
    with open(tsv_file) as tsv_data:
        tsv_lines = tsv_data.readlines()
    with open(ans_file) as ans_data:
        ans_lines = ans_data.readlines()
    corpus = []
    for tsv, ans in zip(tsv_lines, ans_lines):
        corpus.append(parse_mctest_instance(tsv, ans))
    return corpus

def parse_mctest_instance(tsv_chunk, ans_chunk):
    tsv_tabs = tsv_chunk.strip().split('\t')
    ans_tabs = ans_chunk.strip().split('\t')
#    print('ans tabs:', ans_tabs)

    id = tsv_tabs[0]
    ann = tsv_tabs[1]
    passage = tsv_tabs[2]

    # the dictionary for populating a set of passage/questions/answers
    qset_dict = {}
    qset_dict['support'] = [{
        'document': {
            'text': clean_mctest_text(passage)
        }
    }]

    # collect questions / answers
    qset_dict['questions'] = parse_mctest_questions(tsv_tabs[3:], ans_tabs)

    return qset_dict

def parse_mctest_questions(question_list, ans_tabs):
#    print(ans_tabs)
    questions = []
    for i in range(0, len(question_list), 5):
        qdict = {}
        # parse answers
        candidates = []
        correct_answer = ans_tabs[int(i / 5)]
#        print('correct: ', correct_answer)
        for j in range(1,5):
            label = labels[j-1]
#            print('label: ', label)
            answer = {
                'label' : label,
#                'index' : labels.index(label), #label == correct_answer,
                'text' : question_list[i+j]
            }
            candidates.append(answer)
        correct_index = labels.index(correct_answer)
        answer = {
            'index': correct_index,
            'text': question_list[i+correct_index+1]
        }
        # parse question
        qcols = question_list[i].split(':')
        qdict  = {
#            'question-type' : qcols[0],
            'question' : qcols[1],
            'candidates' : candidates,
            'answers': [answer]
        }
        questions.append(qdict)
    return questions

def convert_squad(file_path):
    question_sets = []
    with open(file_path) as data_file:
        data = json.load(data_file)['data']
        for article in data:
            for paragraph in article['paragraphs']:
                qa_set = {
                    'support': [parse_support(paragraph)],
                    'questions': [parse_question(qa_dict) for qa_dict in paragraph['qas']]
                }
                question_sets.append(qa_set)
    return question_sets

def parse_support(para_dict):
    return {
        'document': {
            'text': para_dict['context']
        }
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
        'span': []
#{
#            'start': chosen_answer_start,
#            'end': chosen_answer_start + len(chosen_answer_text)
#        }
    }
    return {
        'question': qa_dict['question'],
        'answers': [result_answer_dict]
    }

def main():
    import sys
    if len(sys.argv) == 3:
        corpus = convert_mctest(sys.argv[1], sys.argv[2])
        print(json.dumps(corpus, indent=2))
#    if len(sys.argv) == 3:
#        with open(sys.argv[2], 'w') as outfile:
#            json.dump(corpus, outfile, indent=2)

if __name__ == "__main__": main()
