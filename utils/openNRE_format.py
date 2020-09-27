import json


def change_span(record):
    text = record['text']
    t = text[:record['h']['pos'][0]].strip().split()
    head_span = [len(t), len(t) + len(text[record['h']['pos'][0]:record['h']['pos'][1]].split())]
    t = text[:record['t']['pos'][0]].strip().split()
    tail_span = [len(t), len(t) + len(text[record['t']['pos'][0]:record['t']['pos'][1]].split())]
    return head_span, tail_span


def change_format(file='../data/demo_nyt10_hard_matched.json'):
    data = json.load(open(file, encoding='utf-8'))
    dc = {}
    for key in data:
        dc[key] = []
        for record in data[key]:
            new_rec = {
                'tokens': record['text'].split(),
                'relation': record['relation'],
                'h': [record['h']['name'], '_', [change_span(record)[0]]],
                't': [record['t']['name'], '_', [change_span(record)[1]]]
            }
            dc[key].append(new_rec)
    return dc


#r = change_format()
#with open('tmp.json', 'w', encoding='utf-8') as f:
#    json.dump(r, f, indent='\t')

