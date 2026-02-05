import os
import pickle
import datetime
from zoneinfo import ZoneInfo 

# This function writes data to a pickle file
# filename can be set by user, otherwise a unique filename will be created using
# current date and time
def pkl_writer(data, output_filename=''):
    if not output_filename:
        os.makedirs('output_pkl', exist_ok=True) 
        # Use current time to create a unique filename 
        now = datetime.datetime.now(ZoneInfo("Europe/London"))
        strf = now.strftime("%Y%m%d%H%M") # Set time format
        output_filename = f'output_pkl/pkl_writer{strf}'
    else: # Ensure folder exists if output_filename is provided manually
        # Extract the directory component of a path
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) 
    # Dump data to a pickle file
    with open(f"{output_filename}.pkl", "wb") as f: 
        pickle.dump(data, f)
    return output_filename

# This function reads a pickle file
def pkl_reader(filename):
    try:
        # Load data from a pickle file
        with open(f"{filename}", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f'Error: The file "{filename}" was not found.')
    except Exception as e:
        print(f'An unexpected error occured: {e}')
    