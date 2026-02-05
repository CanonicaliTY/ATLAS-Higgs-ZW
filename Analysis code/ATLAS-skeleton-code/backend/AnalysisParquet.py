import os
import glob
import time
import datetime
from zoneinfo import ZoneInfo
import re
import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from .ParquetDict import PARQUET_DICT, STR_CODE_COMBO, VALID_STR_CODE # String code and sample filepath

# This function counts total number of events or sum of weights of the data accessed using a string code
def count_num_events(string_code):
    if string_code not in PARQUET_DICT:
        raise ValueError(f'{string_code} not found in PARQUET_DICT.')

    # Use the string code to get the corresponding filepath for parquet files
    read_directory = PARQUET_DICT[string_code]
    if not os.path.isdir(read_directory):
        raise FileNotFoundError(f"Folder '{read_directory}' does not exist")

    # Get all parquet files in the directory as a list
    files = sorted(glob.glob(f'{read_directory}/*.parquet'))
    
    # Use the first parquet file in the list to see if the data has 'totalWeight'
    pq_file = pq.ParquetFile(files[0])
    has_totalWeight = 'totalWeight' in pq_file.schema.names

    # Get total number of events / sum of weights for a given string code
    num_events = 0 # Initialise the number of events / sum of weights
    for file in files:
        pq_file = pq.ParquetFile(file)
        if has_totalWeight: # Get the sum of weigths if the data has the 'totalWeight' column
            num_events += ak.sum(ak.from_parquet(file, columns=['totalWeight'])['totalWeight'])
        else: # Get the number of events if the data doesn't have 'totalWeight'
            num_events += pq_file.metadata.num_rows
    return num_events
    # End of count_num_events() function

# This function parse an input variable
# e.g. input_var = 'lep_pt[0]', this function stacks ('lep_pt[0]', 'lep_pt', '0') with parsed_variables
def parse_var(input_var, parsed_variables):
    if '[' in input_var and ']' in input_var: # Say, if input_var is 'lep_pt[0]'
        try:
            base_var = input_var.split('[')[0] # base_var is 'lep_pt'
            idx_pos_start = input_var.find('[') + 1 # Index starting position in the str
            idx_pos_end = input_var.find(']')
            # Try to convert the value given in '[]' into an int
            index = int(input_var[idx_pos_start : idx_pos_end]) # index is 0
        except Exception as e:
            print(f'Invalid input variable format : {input_var}. '
                  f'Expect "variable" or "variable[int]".\nError: {e}')
            raise
    elif '[' in input_var or ']' in input_var:
        raise ValueError('Expect input variable to be "variable" or "variable[int]".'
                         'Perhaps you forgot a "[" or "]"?')
    else: # No '[]' in input_var
        base_var = input_var
        index = None
    
    return np.vstack([parsed_variables,
                    (input_var, base_var, index)])
# End of parse_var() function

# This function loops through all parquet files for a given directory or string code
# Reads variables based on parsed_variables. Store 'totalWeight' if the column is found in the Parquet file
# Reads files up to a max_num_events calculated in analysis_pq() or read_parquet()
# Is able to write the data read from the parquet files to new parquet files
# Is able to apply selection cuts
# Can choose not to store data in memory
def concatenate_chunks(files, parsed_variables, cut_function, write_parquet, sample_out_dir, max_num_events, return_output):

    sample_data_list = [] # hold data from each file
    chunk_count = 0
    num_events_read = 0
    
    for file in files:
        # Get all columns in the file
        parquet_file = pq.ParquetFile(file)
        all_columns = parquet_file.schema.names
        
        # Update parsed_variables with 'totalWeight' if it is present in the file
        # but not in parsed_variables (to be read from the files and stored)
        if 'totalWeight' in all_columns and 'totalWeight' not in parsed_variables[:, 1]:
            parsed_variables = np.vstack((parsed_variables, ('totalWeight', 'totalWeight', None)))
        # Remove 'totalWeight' from parsed_variables if the column doesn't exist in the file
        elif 'totalWeight' not in all_columns and 'totalWeight' in parsed_variables[:, 1]:
            rows_to_delete = np.any(parsed_variables == 'totalWeight', axis=1)
            parsed_variables = parsed_variables[~rows_to_delete] # to avoid reading and storing a non-existent column
        
        chunk_data_list = [] # Hold data from each row group in one file
        num_row_groups = parquet_file.num_row_groups # Number of row groups in the file
        has_totalWeight = 'totalWeight' in parsed_variables[:, 1] # See if the data is MC

        # Read certain columns from parquet file and store as Awkward arrays row group by row group
        for group in range(num_row_groups):
            if num_events_read >= max_num_events:
                break
            arr = ak.from_parquet(file, columns=parsed_variables[:, 1], row_groups={group})

            # Skip to the next row group if no data found
            if len(arr) == 0:
                print(f'No data found for {parsed_variables[:, 1]} in {file}')
                continue

            # If 'totalWeight' column present in the file, update the num_events using the sum of weights
            # if not all events have totalWeight = 1. Otherwise, just count the number of events
            if has_totalWeight:
                sum_weights = ak.sum(arr['totalWeight'])
                if sum_weights != len(arr): # totalWeight of each event != 1
                    if (num_events_read + sum_weights) > max_num_events: # Can't read all events in this row group
                        weights = ak.to_numpy(arr["totalWeight"])
                        cum_weights = np.cumsum(weights) # Cumulative sum of weight
                        # See where the place the cut off index and slice the array
                        cutoff_idx = np.searchsorted(cum_weights, max_num_events - num_events_read) + 1
                        arr = arr[:cutoff_idx]
                        num_events_read += weights[:cutoff_idx].sum()
                    else: # Can read all events in this row group
                        num_events_read += sum_weights
                else: # totalWeight of each event == 1
                    if (num_events_read + len(arr)) > max_num_events: # Can't read all events in this row group
                        num_events_to_read = max_num_events - num_events_read # Can read these events
                        arr = arr[:num_events_to_read]
                        num_events_read = max_num_events
                    else: # Can read all events in this row group
                        num_events_read += len(arr)
            else: # No 'totalWeight' column, just count number of events
                if (num_events_read + len(arr)) > max_num_events: # Can't read all events in this row group
                    num_events_to_read = max_num_events - num_events_read # Can read these events
                    arr = arr[:num_events_to_read]
                    num_events_read = max_num_events
                else: # Can read all events in this row group
                    num_events_read += len(arr)

            # Selection cut
            if cut_function is not None:
                try:
                    arr = cut_function(arr)
                except Exception as e:
                    print(f'cut_function is a function that takes one argument and returns it.\nException occurred : {e}\n')
                    raise

                # Skip to the next row group if all data has been filtered
                if len(arr) == 0:
                    print(f'No data found for {parsed_variables[:, 1]} in {file} after selection cut')
                    continue

            if write_parquet:
                chunk_data_list.append(arr) # Add data of this row group to write to disk for this file

            # Add any derived field to parsed_variables (these are the ones that will be saved) if not already in it
            for field in arr.fields:
                if field not in parsed_variables[:, 1]:
                    parsed_variables = np.vstack([parsed_variables,
                    (field, field, None)])

            # Loop through variables to validate then save
            for input_var, base_var, index in parsed_variables:
                if base_var not in arr.fields:
                        raise ValueError(f"Variable '{base_var}' not found. Failed to access '{input_var}'. Available variable(s): "
                                   f"{arr.fields}")
                if base_var != input_var:
                    data = arr[base_var]
                    type_str = str(ak.type(data))
                    is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)
            
                    if is_nested: # Variable array is nested
                        # Find out the maximum length of variable array among all events
                        num = ak.num(data)
                        max_num = ak.max(num)
                        index = int(index) # index is str, so convert to int
                        if index >= max_num: # Input index out of range
                            raise IndexError(f'Invalid index for input variable "{input_var}". '
                                             f'Input index should be less than {max_num}.')
                        # If all events have the variable array of same length, no need padding with none
                        # because it takes up a lot memory
                        if not ak.all(num >= index + 1):
                            data = ak.pad_none(data, index + 1, axis=-1)
                        # Add sliced variable to arr as new field
                        arr = ak.with_field(arr, data[:, index], input_var)
                    else: # The array is not nested, but user wants to slice it with [:, index]- raise error
                        raise ValueError(f'{base_var} is not is_nested. Failed to access "{input_var}".')
            # End of loop through variables

            # Remove fields that user doesn't want to save
            unwanted_var = [field for field in arr.fields if field not in parsed_variables[:, 0]]
            if unwanted_var:
                for i in unwanted_var:
                    arr = ak.without_field(arr, i)

            # Add data for this row group to the list that holds data for all files corresponding to a
            # single string code or read_directory
            if return_output:
                sample_data_list.append(arr)
        # End of loop through row groups in one file
        if chunk_data_list:
            if len(chunk_data_list) > 1: # Multiple row groups have data, need concatenation
                chunk_data_ak = ak.concatenate(chunk_data_list)
            else: # Only one row group has data, take the only element in the list
                chunk_data_ak = chunk_data_list[0]

            # Write to parquet file and update chunk_count (for filename)
            ak.to_parquet(chunk_data_ak, f'{sample_out_dir}/chunk{chunk_count}.parquet')
            chunk_count += 1
            
        if num_events_read >= max_num_events:
            break
    # End of loop through all parquet files
    
    if return_output:
        if sample_data_list:
            if len(sample_data_list) > 1: # Concatenate data from all files
                return ak.concatenate(sample_data_list)
            else: # Only one array in the list, so return the only array
                return sample_data_list[0]
    else:
        return None
# End of concatenate_chunks() function


# This function gets a list of parquet files based on string_code_list, then call concatenate_chunks() to process data from each file
def analysis_pq(string_code_list, fraction, parsed_variables, cut_function, write_parquet, output_directory, return_output):
    all_data = {} # Hode data for each entry in string_code_list
    
    for str_code in string_code_list:
        str_code = str(str_code)
        sample_key = str_code

        files = [] # Hold parquet files for this string code
        max_num_events = 0
        if str_code in PARQUET_DICT:
            sample_directory = PARQUET_DICT[str_code]
            pq_files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
            if not pq_files:
                raise FileNotFoundError(f"No .parquet files found with the string code '{str_code}'") 
            files.extend(pq_files) 
            # Update max_num_events with a fraction of total number of events from each file
            max_num_events += count_num_events(str_code) * fraction
        else:
            # For example, str_code may be 'Wlepnu'. It's not in PARQUET_DICT, but in STR_CODE_COMBO
            # as it is actually 'Wenu+Wmunu+Wtaunu' - each of them is in PARQUET_DICT
            if str_code in STR_CODE_COMBO:
                str_code = STR_CODE_COMBO[str_code]

            # If user combine string codes with '+', then get the string code components
            # validate and then add the corresponding parquet files to list
            physics_processes = [code.strip() for code in str_code.split('+')]
            for i in physics_processes:
                if i in PARQUET_DICT:
                    sample_directory =  PARQUET_DICT[i]
                    pq_files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
                    if not pq_files:
                        raise FileNotFoundError(f"No .parquet files found with the string code '{i}'")
                    files.extend(pq_files)
                    # Update max_num_events with a fraction of total number of events from each file         
                    max_num_events += count_num_events(i) * fraction
                else: # String code neither in PARQUET_DICT nor STR_CODE_COMBO
                    raise ValueError(f'Invalid string code: {i}. Available string codes: {VALID_STR_CODE}')

        # Update sample key with fraction used, replace any decimal point to create a valid path
        sample_key = f'{sample_key}_{fraction}'
        sample_key = sample_key.replace('.', '_')
        sample_key = sample_key.replace('+', '_')

        if write_parquet:
            # Create directory to save data to
            sample_out_dir = f'{output_directory}/{sample_key}'
            os.makedirs(sample_out_dir)
        else:
            sample_out_dir = None

        # Process data file by file
        all_data[sample_key] = concatenate_chunks(files, parsed_variables, cut_function,
                                                  write_parquet, sample_out_dir, max_num_events, return_output)
        
    if return_output:
        return all_data
# End of analysis_pq() function

# This function gets a list of parquet files for each subdirectory_names in read_directory,
# then call concatenate_chunks() to process data from each file
def read_parquet(read_directory, subdirectory_names, fraction, parsed_variables, cut_function,
                 write_parquet, output_directory, return_output):
    all_data = {} # Hold data for each subdirectory in read_directory

    # Get all subdirectories name in the read_directory if not provided
    if subdirectory_names is None: 
        subdirectory_names = [name for name in os.listdir(read_directory) 
                              if os.path.isdir(os.path.join(read_directory, name))]
        
    for sample_key in subdirectory_names:
        sample_directory = f"{read_directory}/{sample_key}"
        if not os.path.isdir(sample_directory):
            raise FileNotFoundError(f"Folder '{sample_directory}' does not exist")

        # Get all parquet files in this subdirectory
        files = sorted(glob.glob(f'{sample_directory}/*.parquet'))
        if not files:
            print(f"No parquet files found in directory: {sample_directory}") 

        num_events = 0 # Get total number of events from all files this subdirectory
        for file in files:
            pq_file = pq.ParquetFile(file)
            # Get the sum of weights in the file if 'totalWeight' column exists
            if 'totalWeight' in pq_file.schema.names:
                num_events += ak.sum(ak.from_parquet(file, columns=['totalWeight'])['totalWeight'])
            else: # If 'totalWeight' column doesn't exist, simply count number of events
                num_events += pq_file.metadata.num_rows
                
        max_num_events = num_events * fraction

        # Update sample key with fraction, create valid path by replacing decimal point
        sample_key = f'{sample_key} x{fraction}'
        sample_key = sample_key.replace('.', '_')
        
        if write_parquet:
            sample_out_dir = f'{output_directory}/{sample_key}'
            os.makedirs(sample_out_dir)
        else:
            sample_out_dir = None

        # Process data file by file 
        all_data[sample_key] = concatenate_chunks(files, parsed_variables, cut_function,
                                                  write_parquet, sample_out_dir, max_num_events, return_output)
        
    if return_output:
        return all_data
# End of read_parquet() function
        

# User call this function to read a fraction of data from parquet files
# accessed by string_code_list or read_directory.
# Can apply selection cut; can write the data to disk; can avoid storing data in memory
def analysis_parquet(read_variables, # Read these variables from the files
                     string_code_list=None, # A list of string codes
                     read_directory=None, # Directory to read data from
                     subdirectory_names=None, # Subdirectory names to read from
                     fraction=1, # Fraction of data to read
                     cut_function=None, # A callable that accepts an argument and return it
                     write_parquet=False, # Set to True to write data to parquet files
                     output_directory=None, # Specify the parquet file output location
                     return_output=True # Set to False to not store data in memory (not return the data)
                    ):
    if string_code_list is None and read_directory is None:
        raise ValueError('Either string_code_list or read_directory must be provided.')

    if isinstance(read_variables, str):
        raise TypeError(f'read_variables must be a list. Got a string: {read_variables}')

    time_start = time.time()

    # Parse input variales in read_variables
    parsed_variables = np.zeros((0, 3))
    for input_var in read_variables:
        parsed_variables = parse_var(input_var, parsed_variables)

    if write_parquet:
        # Create a unique output directory name if not provided
        if not output_directory:
            # Use current time to create a unique folder name
            now = datetime.datetime.now(ZoneInfo("Europe/London"))
            strf = now.strftime("%y%m%d%H%M") # Set time format
            output_directory = f'output/analysis_parquet{strf}'
        print(f'Write data to output_directory: {output_directory}')

    # Access data using string_code_list or read_directory by calling analysis_pq() or read_parquet()
    if string_code_list:
        print('Input string_code_list found. Data samples will be accessed by the string code(s).')
        all_data = analysis_pq(string_code_list, fraction, parsed_variables, cut_function, write_parquet, output_directory, return_output)
    elif read_directory:
        print(f'Input read_directory found. Data will be read from {read_directory}.')
        all_data = read_parquet(read_directory, subdirectory_names, fraction, parsed_variables, cut_function, write_parquet, output_directory, return_output)
    # else statement handled at the start of function
        
    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed
    
    if return_output:
        return all_data
# End of analysis_parquet() function
