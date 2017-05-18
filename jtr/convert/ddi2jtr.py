#!/usr/bin/env python3
# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
Convert DDI data to the JTR format.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2017-05-18
'''

from argparse import ArgumentParser, FileType
from collections import defaultdict
from json import dump
from re import compile as re_compile

MEDLINE_RE = re_compile(r'^(?P<path>.*/(?P<id>[0-9]+).*?)(\.grepner)?$')

def main(args):
    def argparser():
        p = ArgumentParser(description='DDI to JTR data converter')
        p.add_argument('mentions', type=FileType('r'), help='DDI mentions')
        p.add_argument('fold', type=FileType('r'), help='DDI fold')
        p.add_argument('output', type=FileType('w'), help='JTR output')
        p.add_argument('-p', '--pretty', action='store_true',
                help='pretty-print JSON')
        return p
    argp = argparser().parse_args(args[1:])

    doc_by_id = {}
    for fpath in (l.split('\t')[0] for l in argp.mentions):
        m_dic = MEDLINE_RE.match(fpath).groupdict()
        id_ = int(m_dic['id'])
        doc_path = m_dic['path']
        doc_path = '/home/ninjin/git/seeds/ddi/' + doc_path # XXX
        if id_ in doc_by_id:
            continue
        with open(doc_path, 'r') as f:
            # Does not preserve sentence boundaries.
            doc_by_id[id_] = f.read().rstrip('\n').replace('\n', ' ')

    pol_by_ddi = {}
    ids_by_ddi = defaultdict(set)
    for l in (l.rstrip('\n') for l in argp.fold):
        pol, d0, d1, chain_s = l.split('\t', maxsplit=3)
        ddi = (d0, d1)
        pol_by_ddi[ddi] = pol
        ids_by_ddi[ddi].update({int(e) for e in chain_s.split('\t')})

    jtr = {
        'meta': 'DDI',
        'globals': {
            'candidates': tuple(),
        },
    }
    instances = []
    for ddi in pol_by_ddi:
        d0, d1 = ddi
        instances.append({
            'support': tuple({'text': doc_by_id[d]} for d in ids_by_ddi[ddi]),
            'questions': ({
                'question': 'Do {} and {} interact ?'.format(d0, d1),
                'candidates': tuple(),
                # Should be global, but no support.
                'answers': ({
                    'text': 'Jawohl!' if pol_by_ddi[ddi] == '+' else 'Nein!',
                }, ),
            },),
        })
    jtr['instances'] = tuple(instances)

    dump(jtr, argp.output, indent=2 if argp.pretty else None)

    return 0

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
