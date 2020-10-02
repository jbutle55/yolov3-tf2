import json
import os
import sys

args = sys.argv
names_file = args[1]
json_file = args[2]

class_list = []
with open(names_file, 'r') as names:
    for line in names:
        class_list.append(line.rstrip())


with open(json_file) as j:
    data = json.load(j)
    for c in data['categories']:
            if c['name'] in class_list:
                continue
            else:
                print(c['name'])
