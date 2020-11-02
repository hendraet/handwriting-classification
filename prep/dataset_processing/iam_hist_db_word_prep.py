import json

import re

infile = 'word_labels_washington.txt'
outfile = 'words_only_washington.json'

goth_chars_regex = re.compile('[a-z0-9]{2,}')

with open(infile, 'r') as f, open(outfile, 'a') as out:
    img_list = []
    for line in f:
        parts = line.split(' ')
        replacements = [('pt', ''), ('-', ''), ('cm', ','), ('s_', ''), ('\n', '')]

        # cleansed_string = re.subn(r'[a-z0-9]{2,}', '§', parts[1])[0]
        cleansed_string = parts[1]
        for replacement in replacements:
            cleansed_string = cleansed_string.replace(replacement[0], replacement[1])
        if all(char.isalpha() for char in cleansed_string):
            # print(parts[0], cleansed_string)
            img_list.append({
                "string": cleansed_string,
                "type": "text",
                "path": "datasets/" + parts[0] + ".png",
            })
            # out.write(parts[0] + ' ' + cleansed_string + '\n')
    print(img_list)
    json.dump(img_list, out, indent=4)
