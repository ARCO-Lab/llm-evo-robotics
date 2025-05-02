import tempfile

def create_temp_xml(robot_xml):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_xml:
        temp_xml.write(robot_xml.encode("utf-8"))
        temp_xml_path = temp_xml.name

    return temp_xml_path

