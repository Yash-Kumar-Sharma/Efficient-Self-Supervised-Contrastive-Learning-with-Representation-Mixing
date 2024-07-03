import sys

def max_num_in_file(filename):
    """Returns the largest integer found in the file"""
    infile = open(filename)
    contents = infile.read()
    infile.close()
    
    lines = contents.splitlines()
    
    maximum = lines[0]
    for line in lines:
        if line > maximum:
            maximum = line
    return maximum  

filename = sys.argv[1]
answer = max_num_in_file(filename)
print(answer)
