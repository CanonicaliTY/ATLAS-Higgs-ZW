import atlasopenmagic as atom
import os
import requests
import re
import uproot
import time
import awkward as ak # for handling complex and nested data structures efficiently
import datetime
from zoneinfo import ZoneInfo
from .EventWeights import WEIGHT_VAR, calculate_weight
atom.set_release('2025e-13tev-beta')
from .DataSetsMagic import VALID_SKIMS, DIDS_DICT

# This function accesses data from local sample files. If not found in sample_path, downloads them
# This function returns a dict (Key: samples' key, Value: corresponding filepath list)
def validate_files(samples, skim, sample_path):
    filepath_dict = {}
    for key, value in samples.items():
        file_path_list = [] # Hold all filepaths
        # value is made using atom.build_dataset, so it is a dict where a key is 'list' and its value
        # is a list of url
        for val in value['list']: 
            # Remove the parent directory path to only get the filename
            fileString = val.split("/")[-1]

            # Validate / Download to the correct folder
            if 'mc' in fileString:
                folder = f'{sample_path}/{skim}/MC'
            else:
                folder = f'{sample_path}/{skim}/Data'

            # Make directory
            os.makedirs(folder, exist_ok=True)
            
            file_path = f'{folder}/{fileString}'
            
            # Download the file if file_path not found
            if os.path.exists(file_path):
                print(f"File {fileString} already exists in {folder}. Skipping download.")
            else:
                print(f"Downloading {fileString} to {folder} ...")

                # if cache=True in get_samples_magic(), we need to remove 'simplecache::' from val to get https://...
                if val.startswith("simplecache::"):
                    val = val.split("simplecache::", 1)[1]
                    
                with requests.get(val, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            file_path_list.append(file_path)
        # End of loop through each file
        filepath_dict[key] = file_path_list
    # End of loop through each samples' key
    return filepath_dict
# End of validate_files() function

# This function builds a dict where keys are string_code_dict's keys and values are urls of sample files
# Those urls can be used to download or access local files via validate_files(), or can be used
# directly by tree.open(url + ': analysis') to stream the files
# Example: string_code_dict = {
#      'Data 2to4lep' : 'Data',
#      'Signal $Z→ee$' : 'Zee',
#      'Signal $Z→μμ$' : 'Zmumu'
# }
def get_samples_magic(skim, string_code_dict, local_files):
    if skim not in VALID_SKIMS:
        raise ValueError(f"'{skim}' is not a valid skim. Valid options are: {VALID_SKIMS}")
    if not isinstance(string_code_dict, dict):
        raise TypeError(f'string_code_dict must be a dict. Got {type(string_code_dict)}')

    samples_defs = {} # Hold dids for each key in string_code_dict
    
    if string_code_dict: # Input dict not empty
        for key, string_codes in string_code_dict.items():
            if not isinstance(string_codes, str):
                raise TypeError('The value of the input string_code_dict must be a str.')
            
            if 'Data' in string_codes:
                samples_defs[key] = {'dids': ["data"]} 
            else:    
                # Split the input string that uses '+' to combine the string code of
                # different physics processes. Strip / remove any white spaces
                physics_processes = [string_code.strip() for string_code in string_codes.split('+')] 
                # string_codes = 'H+ZZllll+ttbar'
                # e.g. physics_processes = ['H','ZZllll','ttbar']

                dataset_id_list = [] # Hold the dataset ids

                for i in physics_processes:
                    if i in DIDS_DICT:    
                        dataset_id_list.extend(DIDS_DICT[i])
                    else:
                        raise ValueError(f'The string code: {i} was not found.')

                samples_defs[key] = {'dids': dataset_id_list}

        # If the list returned by atom.build_dataset is the same as atom.get_urls for each key,
        # then there's no need for the if-else statement here. 
        # if local_files:
        #     samples = atom.build_dataset(samples_defs, skim=skim, 
        #                                  protocol='https',
        #                                  cache=False) # changing cache to True will raise error 
        #                                               # when trying to download the files in validate_files()
        # else:
        #     samples = {}
        #     for key, value in samples_defs.items():
        #         samples[key] = []
        #         for did in value['dids']:
        #             url_list = atom.get_urls(did, skim, protocol='https', cache=True)
        #             samples[key].extend(url_list)
        
        samples = atom.build_dataset(samples_defs, skim=skim, protocol='https', cache=True)
        return samples
# End of get_samples_magic() function

# Calculate the number of events after selection cut. Return the sum of weights for the MC
def calc_sum_of_weights(data):
    if 'totalWeight' in data.fields: # Number of events is given by the weighted count for MC
        return round(ak.sum(data['totalWeight']), 3) # Sum of weights
    else: # The real data
        return len(data)

# Process the data accessed by the filepath/url list for one key in string_code_dict
def process_sample(fraction, luminosity, skim, cut_function, sample_key, 
                   filepath_list, read_variables, save_variables, 
                   write_txt, txt_filename, write_parquet, output_directory, 
                   return_output):
    
    is_Data = 'Data' in sample_key

    # Initialise the number of events before and after selection cut for this key
    # to be written to the txt_filename
    total_num_events_before = 0
    total_num_events_after = 0

    sample_data = [] # Hold data from different files but same sample
    chunk_count = 0 # Count number of chunks in all filestrings
    
    for filestring in filepath_list: # Loop over each file
        
        print(f"\t{filestring} :") 
        
        # Open file
        tree = uproot.open(filestring + ": analysis")

        # Store data for all chunks in this filestring to be concatenated at the end of filestring loop
        file_data = [] 
   
        time_start = time.time()

        keep_fields = [] # Initialise list to hold fields that user wants to save
        keep_fields.extend(save_variables)
        # Loop over data in the tree - each data is a dictionary of Awkward Arrays
        for data in tree.iterate(read_variables, # Read these variables
                                 library="ak", # Return data as awkward arrays
                                 # Process up to a fraction of total number of events
                                 entry_stop=tree.num_entries * fraction):

            # Number of events in this chunk
            number_of_events_before = len(data)
            total_num_events_before += number_of_events_before

            # Apply selection cut
            if cut_function is not None:
                try:
                    data = cut_function(data)
                except Exception as e:
                    print(f'cut_function is a function that takes one argument and returns it.\nException occurred : {e}\n')
                    raise
                    
            # Skip to next chunk if no data       
            if len(data) == 0:
                continue

            # Store Monte Carlo weights
            if 'Data' not in sample_key:
                # Use calculate_weight function from EventWeights.py
                data['totalWeight'] = calculate_weight(data, luminosity, skim)

            # Update keep_fields with derived field (computed in cut_function)
            for field in data.fields:
                if field not in read_variables:
                    keep_fields.append(field)

            # Calculate the number of events after selection cuts
            number_of_events_after = calc_sum_of_weights(data)

            # Validate each field in keep_fields
            for i in keep_fields:
                if i not in data.fields:
                    print(f'Variable "{i}" not found in data - cannot be written to disk.')

            # A list of fields that are not specified in save_variables nor computed in cut_function
            delete_fields = [field for field in data.fields if field not in keep_fields]
           
            if delete_fields: # Remove fields from data as they don't need to be saved
                for i in delete_fields:
                    data = ak.without_field(data, i)
                    
            if return_output:
                # Add the data for this chunk to file_data to be concatenated with
                # other array for this file
                file_data.append(data)

            # Add all events that passed the selection cut for each chunck
            total_num_events_after += number_of_events_after
            
            # Write to disk
            if write_parquet:
                if len(data) != 0:
                    sample_out_dir = f'{output_directory}/{sample_key}'
                    ak.to_parquet(data, f"{sample_out_dir}/chunk_{chunk_count}.parquet")
                    chunk_count += 1

            # Time taken to process up to this chunk (This is cumulative for each file)
            time_elapsed = time.time() - time_start
            
            # Print number of events in this chunck before and after and time elapsed
            print(f"\t\t nIn: {number_of_events_before},"
                  f"\t nOut: \t{number_of_events_after}\t in {round(time_elapsed, 1)} s")
        # End of for loop through chunks of entries in one sample file
        
        if return_output:
            # Stack chunks of data in the same file along the first axis and
            # add to 'sample_data' that holds all the data for one file
            sample_data.append(ak.concatenate(file_data))
    # End of loop through all sample files
    
    if write_txt: # Write summary log
        with open(txt_filename, "a") as f:
            f.write(f'\nSample: {sample_key}\n')
            f.write(f'Total number of input: {total_num_events_before}\n')
            f.write(f'Total number of output: {total_num_events_after}\n')
    if return_output:
        return sample_data
    else:
        return []
# End of process_sample() function

def remove_duplicated_entry(variable_list):
    validated = []
    for var in variable_list:
        if var not in validated:
            validated.append(var)
    return validated
    
# Validate input variables to be read from tree
# Update the variable list with weight-related variables for MC
def validate_read_variables(samples, read_variables, skim):
    # Check if real data and MC are present
    has_Data = any('Data' in key for key in samples)
    has_mc = any('Data' not in key for key in samples)

    validated = remove_duplicated_entry(read_variables)
    
    # For real data, variables to read from database are simply the validated read_variables
    data_read_variables = validated if has_Data else None
    # For MC, weight variables (defined by WEIGHT_VAR in EventWeights) and sum of 
    # weights need to be read too
    mc_read_variables = validated + WEIGHT_VAR[skim] + ["sum_of_weights"] if has_mc else None

    return data_read_variables, mc_read_variables

# This function accesses data based on the key and string codes in string_code_dict
# Example: string_code_dict = {
#      'Data 2to4lep' : 'Data', # Each string code matches a list of dataset ids
#      'Signal $Z→ee$' : 'Zee', 
#      'Signal $Z→μμ$' : 'Zmumu'
# }
# 
def analysis_uproot(skim, # Skim for the dataset.
                          # This parameter is only taken into account when using the 2025e-13tev-beta release.
                    string_code_dict, # A dict which value is a string code
                    luminosity, # Integrated luminosity
                    fraction, # Fraction of data to be read from database
                    read_variables, # Variables to read from database
                    save_variables, # Variables to save in memory or to Parquet files
                    cut_function=None, # A function that accepts an argument and returns it
                    local_files=True, # Access local sample files. Set to False to stream the files
                    sample_path='../backend/datasets', # Path to access or download the local files to
                    write_parquet=False, # Set to True to write data to Parquet files
                    output_directory=None, # Output directory to write Parquet files to
                    write_txt=False, # Set to True to write a summary log in a txt file
                    txt_filename=None, # Filename to write summary log to
                    return_output=True # Set to False to avoid storing data in memory
                   ):
    
    time_start = time.time()

    # Get filepath list if local_files, else get url list for each key
    samples = get_samples_magic(skim, string_code_dict, local_files)
    if local_files:
        # Download sample files if not found in sample_path
        samples = validate_files(samples, skim, sample_path)
    # Uncomment the lines below if you comment out the if-else statement in 
    # get_samples_magic and uncomment the atom.build_dataset line
    else:
        samples = {key : value['list'] for key, value in samples.items()}
   
    if not samples:
        return {} # Empty samples - no analysis needed

    now = datetime.datetime.now(ZoneInfo("Europe/London"))
    
    # Write summary log to a text file
    if write_txt:
        if not txt_filename: # Create text filename if not provided 
            strf = now.strftime("%y%m%d")
            txt_filename = f'txt/analysis_uproot{strf}'
            
        # Make a directory if directory provided doesn't exist
        os.makedirs(os.path.dirname(txt_filename), exist_ok=True)    
        # Write to file current time, luminosity and fraction
        with open(txt_filename, "a") as f:
            f.write('----------------------------------------------------------------\n')
            f.write(f'{now.strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'Luminosity: {luminosity}\nFraction: {fraction}\n')

    # Write to the txt file what variables will be saved
    if write_txt:
        with open(txt_filename, "a") as f:
            f.write(f"Input save_variables: {', '.join(save_variables)} will be saved.\n")

    if write_parquet:
        if not output_directory:
            # Create folder name using luminosity and fraction and current date and time
            output_directory = f"output/lumi{luminosity}_frac{fraction}_"
            strf = now.strftime("%y%m%d%H%M") # Set time format
            output_directory += f'{strf}'
        os.makedirs(output_directory) # Make directory and exist_ok = False
        print(f'\nWrite data to output_directory: {output_directory}\n')
        if write_txt:
            with open(txt_filename, "a") as f:
                f.write(f'Output_directory: {output_directory}\n')

    # Initialise a dict to hold the data for each key
    all_data = {}
    
    # Remove duplicated entry in read_variables and save_variables
    data_read_variables, mc_read_variables = validate_read_variables(samples, read_variables, skim)
    save_variables = remove_duplicated_entry(save_variables)
    
    # Loop over samples
    for sample_key, filepath_list in samples.items():

        if write_parquet:
            os.makedirs(f'{output_directory}/{sample_key}')
    
        if 'Data' in sample_key:
            read_var = data_read_variables
        else:
            read_var = mc_read_variables
        
        # Print which sample is being processed
        print(f'Processing "{sample_key}" samples') 

        # Process data file by file
        sample_data = process_sample(fraction, luminosity, skim, cut_function, sample_key, filepath_list, read_var, save_variables, write_txt, txt_filename, write_parquet, output_directory, return_output)

        if return_output:
            if sample_data: 
                if len(sample_data) > 1: # Concatenate if more than one array for this key
                    sample_data = ak.concatenate(sample_data)
                else: # Use the first and only array if only one array returned by process_sample
                    sample_data = sample_data[0]
                    
                all_data[sample_key] = sample_data # Store array in dict
    
    # Print how much time this function takes
    time_elapsed = time.time() - time_start
    print(f'\n\nElapsed time: {round(time_elapsed, 1)}s')
    
    if return_output:
        return all_data
# End of analysis_uproot() function

