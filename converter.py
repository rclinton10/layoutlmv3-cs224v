import xml.etree.ElementTree as ET
import json

def main():
    # TODO: will eventually want to do this work for a bunch of images (20-30) in a loop
    # so basically move lines 7-38 inside a loop
    xml_file_path = 'data/1.xml'
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data = []
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v2#'}

    for page in root.findall('alto:Layout/alto:Page', ns):
        for print_space in page.findall('alto:PrintSpace', ns):
            for text_block in print_space.findall('alto:TextBlock', ns):
                for text_line in text_block.findall('alto:TextLine', ns):
                    for string in text_line.findall('alto:String', ns):
                        token = string.get('CONTENT')
                        hpos = string.get('HPOS')
                        vpos = string.get('VPOS')
                        width = string.get('WIDTH')
                        height = string.get('HEIGHT')
                        data.append({
                            'token': token,
                            'hpos': hpos,
                            'vpos': vpos,
                            'width': width,
                            'height': height,
                            'ner_tag': 0 # TODO: need to fix this at some point. the XML doesn't have a ner_tags
                        })

    json_data = {
        "id": "1",  # Add a unique ID for the document if needed
        "tokens": [item['token'] for item in data],
        "bboxes": [[int(item['hpos']), int(item['vpos']), int(item['hpos']) + int(item['width']), int(item['vpos']) + int(item['height'])] for item in data],
        "ner_tags": [item['ner_tag'] for item in data],
        "image": "/data/1.jpg"
    }

    output_file_path = 'processed_data.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

main()