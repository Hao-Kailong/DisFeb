import json


def synonym(relation):
    """
    可以用wordnet代替
    也可以用Data Mining工具挖掘规则
    """
    return {
        '/business/company/place_founded' : ['found', 'founder', 'founded', 'create', 'based', 'base'],
        '/location/administrative_division/country': ['country'],
        '/location/country/capital': ['capital'],
        '/people/deceased_person/place_of_death': ['death', 'dead', 'die', 'went away'],
        '/people/person/children': ['children', 'son', 'daughter']
    }[relation]


def hard_match(file='../data/demo_nyt10.json'):
    data = json.load(open(file, encoding='utf-8'))
    dc = {}
    for k, v in data.items():
        for record in v:
            tokens = record['text'].split()
            for w in synonym(k):
                if w in tokens:
                    if k not in dc:
                        dc[k] = [record]
                    else:
                        dc[k].append(record)
                    break
    return dc


#dc = hard_match()
#with open('tmp.json', 'w', encoding='utf-8') as f:
#    json.dump(dc, f, indent='\t')



