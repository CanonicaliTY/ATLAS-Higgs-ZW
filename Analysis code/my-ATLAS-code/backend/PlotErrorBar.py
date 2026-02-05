import numpy as np
import time
import awkward as ak
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import AutoMinorLocator # for minor ticks
import hist
from hist import Hist

def plt_errorbar(main_axes, key, value, xmin, xmax, num_bins, marker):
#  data =
#     label1: {\n"
#     "    'color': str,\n"
#     "    'array':  Array[...],\n"
#     "    'weight':  Array[...]\n"
#     "  }
    bin_edges = np.linspace(xmin, xmax, num_bins + 1)
    bin_centres = (bin_edges[:-1] # left edges
                    + bin_edges[1:] # right edges
                      ) / 2
    
    color = value['color']
    array = value['array']
    weight = value['weight']

    txt = []
    
    # Replace any None with nan then convert to numpy array
    array = ak.to_numpy(ak.fill_none(array, np.nan))

    if weight is not None: # Use storage.Weight() if weight provided by user
        weight = ak.to_numpy(ak.fill_none(weight, 0.0))
        h = Hist.new.Reg(num_bins, xmin, xmax, name=key).Weight()
        h.fill(array, weight=weight)
        view = h.view(flow=False) # 2d array, need unpacking as below
        data_points = view.value # bin values
        data_err = np.sqrt(view.variance)
        # Text annotations
        txt.append(f'- {key}: Weighted Sum (value = {h.sum().value:.3e}, '
                    f'variance = {h.sum().variance:.3e}),')
        txt.append(f'Underflow = {h.view()[0].value:.3e}, '
                   f'Overflow = {h.view()[-1].value:.3e}')
    else: # Use storage.Double() if weight not given by user
        h = Hist.new.Reg(num_bins, xmin, xmax, name=key).Double()
        h.fill(array)
        data_points =  h.view(flow=False) # flat array, no need unpacking
        data_err = np.sqrt(data_points)
        # Text annotations
        txt.append(f'- {key}: Sum (value = {sum(data_points):.3e}),')
        txt.append(f'Underflow = {h.view()[0]:.3e}, '
                   f'Overflow = {h.view()[-1]:.3e}')
        
    # plot the data points with errorbar
    main_axes.errorbar(x=bin_centres, y=data_points, yerr=data_err,
                        marker=marker, color=color, linestyle='none',
                        label=f'{key}')
    return h, txt

# This function calls plt_errorbar for each entry in data_dict
# Example: data_dict = {
# label1 : {array : Array[...], weight : Array[...], color : str},
# label2 : {array : Array[...], weight : None, color : str},
# }
# This function aims to plot different variables on a single figure, or
# plot histograms produced under different selection cut as data points
# with errorbar for better visual comparison
def plot_errorbars(data_dict,
                   xmin, # Left edge of first bin
                   xmax, # Right edge of last bin
                   num_bins, # Number of bins
                   x_label, # x-axis label
                   # Optional arguments start from here
                   y_label='Events',
                   logy=False,
                   title=None,
                   marker='o',
                   title_fontsize=17,
                   label_fontsize=17,
                   legend_fontsize=17,
                   tick_labelsize=15,
                   text_fontsize=14,
                   show_text=False,
                   fig_size=(12, 8)):

    data_format = ("data_dict = {\n"
    "  label1: {\n"
    "    'color': str,\n"
    "    'array':  Array[...],\n"
    "    'weight':  Array[...]\n"
    "  }\n"
    "  label2: {\n"
    "    'color': str,\n"
    "    'array':  Array[...],\n"
    "    'weight': None\n"
    "  }\n"
    "}")

    valid_inner_key = ['array', 'weight', 'color']

    # Validate data_dict
    if not isinstance(data_dict, dict):
       raise TypeError(f"Expect 'data_dict' (the first positional argument of the function) to be a dict. The correct format is:\n{data_format}")

    for key, value in data_dict.items():
        if not isinstance(value, dict):
            raise TypeError(f'Expect the value of the inner dict to be a dict. Got {type(value)}')
        if len(value) != 3:
            raise ValueError(f'Expect the value of "{key}" to be a dict with three keys: ["array", "weight", "color"]. Got {len(value)} keys instead')
        for inner_key in value:
            if inner_key not in valid_inner_key:
                raise KeyError(f'Invalid inner key "{inner_key}" in {key} dict. Valid inner key: ["array", "weight", "color"]')
        array = value['array']
        weight = value['weight']
        
        if isinstance(array, str):
            raise TypeError(f'{key} dict : The value of the inner key "array" should be an array. Got a str instead')
        if weight is not None:
            if isinstance(weight, (str, int, float)):
                raise TypeError(f'{key} dict : The value of the inner key "weight" should be an array or None. Got a {type(weight)} instead')
            if len(array) != len(weight):
                raise ValueError(f'{key} dict : The value of the inner key "weight" should be None or an array with the same length as "array". Got {len(weight)} and {len(array)}')

    # Validate xmin and xmax
    if not isinstance(xmin, (int, float)):
        raise TypeError(f'x_min must be a number. Got "{x_min}"')
    if not isinstance(xmax, (int, float)):
        raise TypeError(f'x_max must be a number. Got "{x_max}"')
    if xmax <= xmin:
        raise ValueError(f'x_max must be greater than x_min. Got x_max = "{x_max}", x_min = "{x_min}"')

    # Validate num_bins
    if not isinstance(num_bins, int):
        raise TypeError(f"num_bins needs to be an int. Got '{num_bins}' instead")
    if num_bins < 2:
        raise ValueError(f"num_bins needs to be greater than 1. Got '{num_bins}' instead")

    # Validate fig_size
    if (not isinstance(fig_size, tuple)
        or not all(isinstance(value, (int, float)) for value in fig_size)):
        raise ValueError("fig_size must be a tuple of two numbers.")
    if len(fig_size) != 2 or not all(value > 0 for value in fig_size):
        raise ValueError("fig_size must be a tuple of two positive numbers.")
        
    time_start = time.time()
    
    fig = plt.figure(figsize=fig_size)  # Create empty figure
    main_axes = plt.gca()

    text = [] # Hold text info for each histogram
    hists_list = [] # Hold histogram objects

    # Plot data points with errorbar for each entry in data_dict
    for key, value in data_dict.items():
        h, txt = plt_errorbar(main_axes, key, value, xmin, xmax, num_bins, marker)
        text.extend(txt)
        hists_list.append(h)

    # Text annotations under the plot
    if show_text:
        for i, line in enumerate(text):
            main_axes.text(-0.05, -0.15 - i * 0.05, line, ha='left', va='top', transform=main_axes.transAxes, fontsize=text_fontsize)
            
    # separation of x axis minor ticks
    main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          labelsize=tick_labelsize, # Label size
                          top=True, # draw ticks on the top axis
                          right=True) # draw ticks on right axis

    main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                         x=1, horizontalalignment='right' )
    main_axes.set_ylabel(y_label, fontsize=label_fontsize,
                         y=1, horizontalalignment='right') 
    main_axes.legend(frameon=False, fontsize=legend_fontsize)
    if title is not None:
        main_axes.set_title(title, fontsize=title_fontsize)
    
    if logy:
        main_axes.set_yscale('log')

    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    plt.tight_layout()
    plt.show()

    return fig, hists_list