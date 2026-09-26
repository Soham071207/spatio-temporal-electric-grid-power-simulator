import urllib.request, json

headers = {'x-api-key': 'a7daa2e7130d36c8d1ccd0133280d7b22c0f3e7df2b260fbc15e7c0ccb8e1a59'}

for ind_id in [2452, 2453, 10521, 10339]:
    req = urllib.request.Request(
        f'https://api.esios.ree.es/indicators/{ind_id}?start_date=2024-03-01T00:00:00&end_date=2024-03-31T23:59:59',
        headers=headers
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    vals = data['indicator']['values']
    name = data['indicator']['name']
    andalucia = [v for v in vals if v['geo_id'] == 4]
    print(f'[{ind_id}] {name}')
    print(f'  Total values: {len(vals)}, Andalucia values: {len(andalucia)}')
    if andalucia:
        dt = andalucia[0]['datetime'][:16]
        val = andalucia[0]['value']
        print(f'  First: {dt} = {val}')
        if len(andalucia) > 1:
            dt2 = andalucia[1]['datetime'][:16]
            val2 = andalucia[1]['value']
            print(f'  Second: {dt2} = {val2}')
    print()
