import xml.etree.ElementTree as ET
import os
from datasets import Dataset, DatasetDict, Features, Sequence, Split, ClassLabel, Value, Image as HFDatasetImage
from sklearn.model_selection import train_test_split

NUM_IMAGES = 2 # the # of FD newspaper images (and XMLs) we've downloaded for processing

def main():
    output = []
    for i in range(1, NUM_IMAGES + 1):
        xml_file_path = f"data/{i}.xml"
        print(f"Parsing {xml_file_path}...")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        data = []
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v2#'}

        # Parsing XML data
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
                                'hpos': int(hpos),
                                'vpos': int(vpos),
                                'width': int(width),
                                'height': int(height),
                                'ner_tag': 0  # Placeholder for NER tag
                            })

        image_path = os.path.join("data", f"{i}.jp2")
        # Load image as file path (instead of object) to avoid shape issue
        entry = {
            "id": str(i),
            "tokens": [item['token'] for item in data],
            "bboxes": [[item['hpos'], item['vpos'], item['hpos'] + item['width'], item['vpos'] + item['height']] for item in data],
            "ner_tags": [item['ner_tag'] for item in data],
            "image": image_path  # Using file path for Image compatibility
        }
        output.append(entry)

    # Split the data into train and test sets (e.g., 80% train, 20% test)
    train_data, test_data = train_test_split(output, test_size=0.2, random_state=42)

    # Define the dataset features
    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "bboxes": Sequence([Value("int32")]),
        "ner_tags": Sequence(ClassLabel(num_classes=2, names=["O", "LABEL"])), # Example labels
        "image": HFDatasetImage() # Use HF Image feature to store images
    })

    # Create Dataset from split data
    train_dataset = Dataset.from_list(train_data, features=features)
    test_dataset = Dataset.from_list(test_data, features=features)

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset_dict.push_to_hub("rclinton/224v")
    print("Just pushed to hub and can be viewed at URL https://huggingface.co/datasets/rclinton/224v")

main()