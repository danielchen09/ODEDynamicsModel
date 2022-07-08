import json

def get_filename(filepath):
    return filepath.split('/')[-1]

def change_extention(filepath, ext):
    filepath = filepath.split('.')[0]
    return filepath + '.' + ext

def get_unimal_metadata(xml_path):
    rootdir = '/'.join(xml_path.split('/')[:-2])
    return read_json(f'{rootdir}/metadata/{change_extention(get_filename(xml_path), "json")}')

def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)