import json

# Opening JSON file
f = open('../annotations.json')

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
for key in data:
    print(key)

for value in data['annotations']:
    print(value)

# Closing file
f.close()
