# <form created="2002-01-14" height="3542" id="a01-000u" last-modified="2002-07-17" skew="287" status="final" version="3.0_beta" width="2479" writer-id="000">
#     <machine-printed-part>
#         <machine-print-line text="A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers is to"/>
#         <machine-print-line text="be made at a meeting of Labour M Ps tomorrow. Mr. Michael Foot has put down"/>
#         <machine-print-line text="a resolution on the subject and he is to be backed by Mr. Will Griffiths, M P for"/>
#         <machine-print-line text="Manchester Exchange."/>
#     </machine-printed-part>
#     <handwritten-part>
#         <line ass="-187" asx="0" asy="739" character-width="999" dss="-187" dsx="0" dsy="831" fd0="12129" fd1="2994" fd2="-339" filter-width="5" id="a01-000u-00" lbs="-108" lbx="0" lby="810" segmentation="ok" slant="90000" stroke-width="7333" threshold="154" text="A MOVE to stop Mr. Gaitskell from" ubs="-347" ubx="0" uby="769">
#             <word id="a01-000u-00-00" sentence-start="yes" tag="AT" text="A">
#                 <cmp x="408" y="768" width="27" height="51"/>
#             </word>
#             <word id="a01-000u-00-01" tag="NN" text="MOVE">
#                 <cmp x="507" y="768" width="63" height="46"/>
#                 <cmp x="568" y="770" width="56" height="41"/>
#                 <cmp x="631" y="768" width="38" height="41"/>
#                 <cmp x="676" y="772" width="31" height="36"/>
#             </word>
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
                    # print(image_path, text)
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



