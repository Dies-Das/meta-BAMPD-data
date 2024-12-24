# Define the file paths
file1 = 'cpp.txt'  # File containing tuples like (3,1,0,1,)
file2 = 'python.txt'  # File containing tuples like (0, 3, 0, 3)

# Helper function to read and parse tuples from a file
def read_tuples(file_path):
    with open(file_path, 'r') as file:
        tuples = set()
        for line in file:
            # Strip whitespace and remove trailing commas if present
            clean_line = line.strip().rstrip(',')
            clean_line = clean_line.rstrip(';')
            # Convert the string to a tuple of integers
            tuple_value = eval(clean_line)
            tuples.add(tuple_value)
    return tuples

# Read tuples from both files
tuples_in_file1 = read_tuples(file1)
tuples_in_file2 = read_tuples(file2)

# Find tuples in file2 that are not in file1
unique_to_file2 = tuples_in_file2 - tuples_in_file1

# Output the result
print("Tuples in file2 but not in file1:")
for t in unique_to_file2:
    print(t)