from turtle import st
from lxml import etree
from lxml.etree import Element
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import json
from glob import glob

def replace(file_path, search_str, replace_fn):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if search_str in line:
                    new_file.write(replace_fn(line))
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def uncomment(line):
    start, end = '<!--', '-->'
    if not start in line or not end in line:
        return line
    start_idx = line.index(start)
    prefix = line[:start_idx]
    line = line[start_idx:].replace(start, '').replace(end, '').strip() + '\n'
    return prefix + line

def comment(line):
    start, end = '<!-- ', ' -->'
    if start in line and end in line:
        return line
    prefix = line[:line.index('<')]
    content = line.strip()
    return prefix + start + content + end + '\n'


xml_paths = glob('../unimals_100/train/xml/*.xml')
for xml_path in xml_paths:
    replace(xml_path, 'name="floor"', uncomment)