

def read_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_content(file_path, content):
    with open(file_path, 'w') as file:
        file.writelines(content)
        # file.write(content)


def find_lines_with_string(file_path, search_string):
    lines_found = []  # List to store the results

    # Open the file and read it line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        line_number = 0
        for line in file:
            if search_string in line:
                lines_found.append((line_number, line))
            line_number += 1

    return lines_found

