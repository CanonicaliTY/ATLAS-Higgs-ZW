import hist
from hist import Hist 
import awkward as ak
import numpy as np

# User provide data, number of bins, bin range (xmin, xmax) and histogram name (an arg for Hist.new.Reg)
# This function makes Hist object using the user input, and returns the bin values, variances (if any), and bin centres
# If weight is None, storage .Double() will be used; if provided, .Weight() will be used
def get_histogram(variable_data, num_bins, xmin, xmax, hist_name, weight=None):
    # Replace any None with np.nan, and then convert to numpy
    variable_data = ak.to_numpy(ak.fill_none(variable_data, np.nan))
    if weight is not None:
        if isinstance(weight, (str, int, float)):
            raise TypeError(f'weight found to be {type(weight)}. It should be None or an Awkward Array that has the same length as variable_data') 
        if len(weight) != len(variable_data):
            raise ValueError(f'weight has to have the same length as variable_data. Got {len(weight)} and {len(variable_data)}')
        
        weight = ak.to_numpy(ak.fill_none(weight, 0.0))
        h = Hist.new.Reg(num_bins, xmin, xmax, name=hist_name).Weight()
        h.fill(variable_data, weight=weight)
        view = h.view(flow=False)
        value = view.value
        variance = view.variance
    else:
        h = Hist.new.Reg(num_bins, xmin, xmax, name=hist_name).Double()
        h.fill(variable_data)
        value = h.view(flow=False)
        variance = None
        
    bin_centres = h.axes[0].centers
    
    return value, variance, bin_centres