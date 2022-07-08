from lxml import etree


def parse_xml(xml_path):
    parser = etree.XMLParser(remove_blank_text=True)
    return etree.parse(xml_path, parser)

def tree_traversal(etree, reversed=-1):
    order = []
    root = etree.findall('./worldbody')[0]
    def _tree_traversal(node):
        order.append(node)
        children = [x for x in node.iterfind('./body')]
        if reversed:
            children = children[::-1]
        for child in children:
            _tree_traversal(child)
    _tree_traversal(root)
    return order[1:] # exclude worldbody

def find_child_by_tag(etree, tag):
    return [x for x in etree.iterfind('./' + tag)]

def get_property(xml_element, prop_name):
    for item in xml_element.items():
        if item[0] == prop_name:
            return item[1]
    return None