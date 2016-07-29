import sys
import json
import re
from time import sleep
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def annotate_candidates(data, tag_seqs):
    num_cands_found = 0
    for instance in data:
        support_text = instance['support'][0]['text']
        support_tokens = instance['support'][0]['tokens']
        support_postags = instance['support'][0]['postags']
        cand_token_ids = get_candidates(support_postags, tag_seqs)
        cands = []
        for s,e in cand_token_ids:
            cand_char_start = support_tokens[s][0]
            cand_char_end = support_tokens[e][1]
            cand_dict = {
                'text': support_text[cand_char_start:cand_char_end],
                'span': [cand_char_start, cand_char_end]
            }
            cands.append(cand_dict)
        for question in instance['questions']:
            question['candidates'] = cands
        num_cands_found += len(cands)
    sys.stderr.write('Found on average {0:.1f} candidates per question.\n'.format(num_cands_found / len(data)))
    return data

def get_candidates(support_postags, tag_seqs):
    cands = []
    for i, tag in enumerate(support_postags):
        for ts in tag_seqs:
            ts_len = len(ts)
            for j in range(0, ts_len):
                if ts[ts_len-j-1] != support_postags[i-j]:
                    break
                if j == ts_len-1:
                    cands.append([i-j, i])
    return cands


def get_candidate_postags(data, top_k=300):
    pos2counts = {}
    for instance in data:
        for question in instance['questions']:
            support_text = instance['support'][0]['text']
            support_char_offsets = instance['support'][0]['tokens']
            support_postags = instance['support'][0]['postags']
            for answer in question['answers']:
                astart,aend = answer['span']
                token_start_id = token_start_idx_from_char_offset(astart, support_char_offsets)
                token_end_id = token_end_idx_from_char_offset(aend, support_char_offsets)
                answer_postag_str = ' '.join(support_postags[token_start_id:token_end_id+1])
                if answer_postag_str in pos2counts:
                    pos2counts[answer_postag_str] += 1
                else:
                    pos2counts[answer_postag_str] = 1
    pairs = sorted(pos2counts.items(), key=lambda x: -1 * x[1])
    total = sum(pos2counts.values())
    tally = 0
    for k,v in pairs[0:top_k]:
        tally += v
    sys.stderr.write('Using k={0}, there is {1:.2f}% coverage on the train data.\n'.format(top_k, tally/total))
    return [k.split() for k,v in pairs[0:top_k]]


def token_start_idx_from_char_offset(char_offset, token_offsets):
    for tidx, to in enumerate(token_offsets):
        if char_offset < to[1]:
            return tidx
    print('Incorrect char offset {} into token offsets {}'.format(char_offset, token_offsets))
    sys.exit()
    return -1

def token_end_idx_from_char_offset(char_offset, token_offsets):
    for tidx, to in enumerate(token_offsets):
        if char_offset <= to[1]:
            return tidx
    print('Incorrect char offset {} into token offsets {}'.format(char_offset, token_offsets))
    sys.exit()
    return -1

def main():
    import sys
    if len(sys.argv) == 2:
        data = read_data(sys.argv[1])
        tag_seqs = get_candidate_postags(data)
        data_with_cands = annotate_candidates(data[0:1], tag_seqs)
        print(json.dumps(data, indent=2))

if __name__ == "__main__": main()
