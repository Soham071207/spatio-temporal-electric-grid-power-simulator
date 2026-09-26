import urllib.request, re

html = open('apidata.html', 'r', encoding='utf-8').read()
scripts = re.findall(r'src="(.*?\.js.*?)"', html)
for script in scripts:
    if script.startswith('/'):
        script = 'https://www.ree.es' + script
    try:
        data = urllib.request.urlopen(script).read().decode('utf-8', errors='ignore')
        if '8741' in data or 'regionSelector' in data:
            print("Found in", script)
            # print snippet
            idx = data.find('regionSelector')
            if idx == -1: idx = data.find('8741')
            print(data[max(0, idx-100):idx+500])
    except Exception as e:
        print("Failed to download", script, e)
