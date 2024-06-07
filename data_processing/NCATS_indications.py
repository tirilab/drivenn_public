import requests
from datetime import datetime
import pandas as pd

def get_additional_data(id):
    r = requests.get(f'https://drugs.ncats.io/api/v1/substances({id})/@additional')
    if 200 == r.status_code:
        return r.json()
    return None

for facet, name in zip(['Condition/Congestive heart failure', 'Condition/Myocardial infarction', 'Condition/Coronary artery disease'], ["CHF", "MI", "CAD"]):
    params = {'facet': [facet, 'Substance Form/Principal Form'], 'top': 100, 'skip': 0 }
    export_file = f'data/NCATS_exports/export_all_uid_{name}.tsv'
    
    print("Searching NCATS for " + facet)
    
    with open(export_file, "w") as f:
        f.write('UNII\tName\tsmiles\tYear\tConditions\tTargets\tStatus\tSource\tSources\tSourceIds\tUIDS\n')
        total = 0
        while True:
            r = requests.get('https://drugs.ncats.io/api/v1/substances/search', params)
            if 200 != r.status_code:
                print(f'{r.url} returns status code {r.status_code}!')
                break
    
            json = r.json()
            count = json['count']
            total = total + count
            
            for r in json['content']:
                additional_data = get_additional_data(r['uuid'])
                if 'structure' in r.keys():
                    smiles = r['structure']['smiles']
                else:
                    if 'polymer' in r.keys():
                        smiles = r['polymer']['displayStructure']['smiles']
                    else:
                        print("no smiles found")
                        smiles = ""
                if additional_data != None:
                    status = None
                    year = None
                    conditions = []
                    targets = []
                    sources = []
                    s_ids = []
                    u_ids = []
    
                    # get Approval Status, Approval Year, Conditions, Targets, and Sources
                    for d in additional_data:
                        if (d['name'] == 'Highest Development Event'
                            or (d['name'] == 'Earliest Approved Event' and not status)
                            ):
                            status = d['value']
                        elif d['name'] == 'Approval Year':
                            year = d['value']
                        elif d['name'] == 'Conditions':
                            try:
                                conditions.append(d['value']['label'])
                            except:
                                print(d)
                        elif d['name'] == 'Targets':
                            try:
                                targets.append(d['value']['label'])
                            except:
                                print(d)
                        elif d['name'] == 'Sourcing':
                            sources.append(d['value']["sourcing_vendor_name"])
                            s_ids.append(d['value']["sourcing_vendor_substance_id"])
                        elif d['name'] == 'PubMed':
                            try:
                                u_ids.append(d['value']['uid'])
                            except:
                                print(d)
                        
    
                    # if status is available, format the values accordingly and write to file  
                    if status and 'sourceURL' in status.keys():
                        f.write('\t'.join([r['approvalID'],
                                           r['_name'],
                                           smiles,
                                           year,
                                           ';'.join(conditions),
                                           ';'.join(targets),
                                           status['status'],
                                           status['sourceURL'],
                                           ';'.join(sources),
                                           ';'.join(s_ids),
                                           ';'.join(u_ids),
                                           ]) + '\n'
                                )
            
            if count < json['top']:
                break
            params['skip'] = total