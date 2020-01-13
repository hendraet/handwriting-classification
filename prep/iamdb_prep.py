import json

import os
import xml.etree.ElementTree as ET


def get_numbers(infile):
    mapping = []
    dir_prefix = 'datasets/'

    tree = ET.parse(infile)
    root = tree.getroot()
    image_dir = root.attrib['id']
    lines = root[1]

    for line in lines:
        for word in line:
            if word.tag == 'word':
                image_path = image_dir.split('-')[0] + '/' + image_dir + '/' + word.attrib['id'] + '.png'
                text = word.attrib['text']
                if all(char.isdigit() for char in text):
                    mapping.append({
                        'string': text,
                        'type': 'num',
                        'path': os.path.join(dir_prefix, image_path)
                    })
    return mapping


# There's a difference between xml files ending in x, u and without ending but I can't figure out what it is and the
# documentation is shitty.

xml_dir = '../datasets/iamdb/xml/'
out_file = 'dataset_descriptions/numbers_iamdb.json'
mappings = []

for xml_file_path in os.listdir(xml_dir):
    mappings.extend(get_numbers(xml_dir + xml_file_path))

with open(out_file, 'w') as out:
    json.dump(mappings, out, indent=4)



