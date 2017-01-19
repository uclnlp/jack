import sys
import json

def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def tree_stats(data):
    span2counts = {}
    for instance in data:
        for question in instance['questions']:
            support_text = instance['support'][0]['text']
            support_char_offsets = instance['support'][0]['tokens']
            support_postags = instance['support'][0]['postags']
            for answer in question['answers']:
                astart,aend = answer['span']
                token_start_id = token_start_idx_from_char_offset(astart, support_char_offsets)
                token_end_id = token_end_idx_from_char_offset(aend, support_char_offsets)
                sent_start_id =
                sent_end_id =
                # answers which cross sent boundaries stats
                # if same sentence, get that sentence's tree, what constituent corresponds to answer?

def pos_stats(data):
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
    count = 0
    print('Top    Freq      %     PosTag Sequence')
    print('--------------------------------------')
    for k,v in pairs:
        tally += v
        count += 1
        print('{0:<5} {1:>5}    {2:>4.1f}    {3}'.format(count, v, 100.0 * tally / total, k))

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
        pos_stats(data)

if __name__ == "__main__": main()
