import csv
import json
import sys
from collections import Counter

input_fn = sys.argv[1]
output_fn = sys.argv[2]

dataset = []
squad_style_dataset = {"data": dataset, "version": "1"}

with open(input_fn, "r") as f:
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        [story_id, question, answer_char_ranges, is_answer_absent, is_question_bad, validated_answers, story_text] = row

        spans = None
        if validated_answers:
            answers = json.loads(validated_answers)
            spans = [k for k, v in answers.items() if ":" in k]
        else:
            answers = Counter()
            for rs in answer_char_ranges.split("|"):
                for r in set(rs.split(",")):
                    if ":" in r:
                        answers[r] += 1
            spans = [k for k, v in answers.items() if ":" in k and v >= 2]

        if spans:
            example = {"title": story_id, "paragraphs": [
                {
                    "context": story_text,
                    "qas": [{
                        "question": question,
                        "id": story_id + "_" + question.replace(" ", "_"),
                        "answers": [{
                                        "answer_start": int(span.split(":")[0]),
                                        "text": story_text[int(span.split(":")[0]):int(span.split(":")[1])]
                                    } for span in spans]
                    }]
                }
            ]}
            dataset.append(example)
            # else:
            #    print("No span found for %s" % story_id)

with open(output_fn, "w") as f:
    json.dump(squad_style_dataset, f)
