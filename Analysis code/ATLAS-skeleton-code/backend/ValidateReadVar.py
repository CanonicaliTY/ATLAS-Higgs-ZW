import os
import glob
import pyarrow.parquet as pq
from .ParquetDict import PARQUET_DICT, STR_CODE_COMBO, VALID_STR_CODE

# Get valid variables for a given string code. If string codes combined with '+',
# the string codes will be split and used to get the fields of its first parquet file 
# (the valid variables). Data in all parquet files should have the same fields
def get_valid_variables(string_code):
    if not isinstance(string_code, str):
        raise TypeError(f'string_code must be a str. Got {type(string_code)}')
    if string_code not in PARQUET_DICT:
        if string_code in STR_CODE_COMBO:
            string_code = STR_CODE_COMBO[string_code]
        # Split the string codes that are combined with '+'
        physics_processes = [code.strip() for code in string_code.split('+')]
        # Use the first string code to return valid variable list
        string_code = physics_processes[0] 
        print(f'Validate variables using the string code {string_code}')
        
    if string_code not in PARQUET_DICT:
        raise ValueError(f'{string_code} not found. Available string codes: {VALID_STR_CODE}')
        
    read_directory = PARQUET_DICT[string_code]
    if not os.path.isdir(read_directory):
        raise FileNotFoundError(f"Folder '{read_directory}' does not exist")

    # Get parquet files for string_code
    files = sorted(glob.glob(f'{read_directory}/*.parquet'))
    print(f'Variables validated using {files[0]}')

    # Use the schema of the first parquet file to give valid variable list
    parquet_file = pq.ParquetFile(files[0])
    schema = parquet_file.schema.to_arrow_schema()
    var_list = [field.name for field in schema]
    return var_list
        

# Return list of variables that are in read_variables and are valid for
# all entries in string_code_list
def validate_read_variables(string_code_list, read_variables):
    validated = []
    
    for string_code in string_code_list:
        valid_var_list = get_valid_variables(string_code)

        # Skip invalid and duplicated entry
        for variable_input in read_variables:
            if variable_input not in valid_var_list:
                print(f"Skipping '{variable_input}' - invalid input for string code '{string_code}'")
            elif variable_input in validated:
                continue
            else:
                validated.append(variable_input)
    return validated
