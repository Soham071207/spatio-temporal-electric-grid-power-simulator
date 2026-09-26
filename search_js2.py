import urllib.request
data = urllib.request.urlopen('https://www.ree.es/themes/custom/ree/js/ree.js?tigpxm').read().decode('utf-8', errors='ignore')
idx = data.find('regionSelector.addEventListener("change"')
print(data[idx:idx+1500])
