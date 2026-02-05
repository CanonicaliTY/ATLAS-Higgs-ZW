import re
import time
import awkward as ak
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import AutoMinorLocator # for minor ticks
import hist
from hist import Hist 

# This function returns a flat Awkward array for a variable using dict key and value input
def get_variable_data(variable, key, value, valid_var):
    if variable in valid_var:
        variable_data = value[variable] # Expect a flat array
        # Check if the array is nested
        type_str = str(ak.type(variable_data))
        # Raise error if array is nested
        if "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str):
            raise ValueError(f'Invalid input variable format : {variable}. '
                             f'Expect "{variable}[int]".')
    else:
        # '[]' given in input. e.g. lep_pt[0]
        if '[' in variable and ']' in variable:
            base_var = variable.split('[')[0] # this will be 'lep_pt'
            try:
                # Index starting and ending position in the str
                idx_pos_start = variable.find('[') + 1 
                idx_pos_end = variable.find(']')
                # Extract the '0' from 'lep_pt[0]'
                index = variable[idx_pos_start : idx_pos_end]
                index = int(index) # Try converting str to int
            except Exception as e:
                print(f'Invalid input variable format : "{variable}". '
                      f'Expect "variable" or "variable[int]".\nError: {e}')

            # Check if this is a nest array
            variable_data = value[base_var]
            type_str = str(ak.type(value[base_var]))
            is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)
            if is_nested:
                # Find out the maximum length of variable array among all events
                num = ak.num(variable_data)
                max_num = ak.max(num)
                if index >= max_num: # Index out of range
                   raise IndexError(f'Invalid index for input variable "{variable}". '
                                    f'Input index should be less than {max_num}.')
                # If all events have the variable array of same length, no need padding with none
                # because it takes up a lot memory
                if not ak.all(num >= index + 1):
                    data = ak.pad_none(data, index + 1, axis=-1)

                variable_data = value[base_var][:, index] # Slice the array
            else: # Raise error if user wants to slice a flat array with [:, index]
                raise ValueError(f'Invalid input variable format : "{variable}". '
                                 f'Did you mean "{base_var}"?')
        # End of if statement (if '[' and ']' present in input variable)
        elif '[' in variable or ']' in variable:
            raise ValueError('Expect input variable to be "variable" or "variable[int]".'
                             'Perhaps you forgot a "[" or "]"?')
        else:
            raise ValueError(f"Variable '{variable}' not found in sample '{key}'. Available variable(s):"
                       f"{valid_var}")
    return variable_data
# End of get_variable_data() function

# This function plots data points if 'Data' in the key, 
# plots stacked histogram on top of background if 'Signal' in the key
# Can plot stacked histograms for multiple Signal and Background entries
# Example, data_dict = {
# 'Data' : Ak.array(...),
# 'Signal' : Ak.array(...),
# 'Background' : Ak.array(...)
#}
def stacked_histogram(data_dict, color_list, variable, xmin, xmax, 
                      num_bins, main_axes, marker, show_back_unc, 
                      residual_axes, x_label, label_fontsize, 
                      tick_labelsize, residual_plot_ylim):
    
    background_x = [] # hold the MC background histogram entries
    background_weights = [] # hold the MC background weights
    background_colors = [] # hold the colors of the MC bars
    background_labels = [] # hold the legend labels of the MC bar

    # Similarly, for MC signal
    signal_x = []
    signal_weights = []
    signal_colors = []
    signal_labels = []

    # Whether there is signal and background in data_dict
    signal_input = False
    background_input = False

    bin_edges = np.linspace(xmin, xmax, num_bins + 1)
    bin_centres = (bin_edges[:-1] # left edges
                    + bin_edges[1:] # right edges
                      ) / 2
    widths = (bin_edges[1:] # left edges
                - bin_edges[:-1]) # right edges

    hists = [] # hold Hist object for each key
    text = [] # hold histogram info for each key
    bullets = 1 # for text annotations

    for (key, value), color in zip(data_dict.items(), color_list):
        if isinstance(value, dict):
            valid_var = list(value.keys())
        elif isinstance(value, ak.Array):
            valid_var = value.fields
        else:
            print(f'Key "{key}" : Unexpected type of dict value. Expect dict or Awkward Array.')
            raise TypeError

        # Validate the input variable
        variable_data = get_variable_data(variable, key, value, valid_var)

        # Plot data points
        if 'Data' in key:
            hist_data = Hist.new.Reg(num_bins, xmin, xmax, name=key).Double()
            # Fill values, replacing any None with nan first
            hist_data.fill(ak.to_numpy(ak.fill_none(variable_data, np.nan)))
            # For text annotation
            text.append(f'({bullets}) {key}: Sum (value = {sum(hist_data.view(flow=False)):.3e}),')
            text.append(f'Underflow = {hist_data.view(flow=True)[0]:.3e}, Overflow = {hist_data.view(flow=True)[-1]:.3e}\n')
            bullets += 1
            
            data_x = hist_data.view(flow=False) # histogram bin values
            data_x_errors = np.sqrt(data_x) # statistical error on the data
            hists.append(hist_data)
            
            # Plot the data points
            main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                marker=marker, color=color, linestyle='none',
                                label=key) 
        elif 'Signal' in key:
            signal_x.append(ak.to_numpy(ak.fill_none(variable_data, np.nan)))
            if 'totalWeight' in valid_var:
                signal_weights.append(ak.to_numpy(value['totalWeight']))
            else: # assume totalWeight = 1 if 'totalWeight' not in field
                signal_weights.append(np.ones(len(variable_data)))
            signal_colors.append(color) # bar color
            signal_labels.append(key) # legend label
            signal_input = True # MC signal present
        else:
            background_x.append(ak.to_numpy(ak.fill_none(variable_data, np.nan)))
            if 'totalWeight' in valid_var:
                background_weights.append(ak.to_numpy(value['totalWeight']))
            else: # assume totalWeight = 1 if 'totalWeight' not in field
                background_weights.append(np.ones(len(variable_data)))
            background_colors.append(color)
            background_labels.append(key)
            background_input = True # Background signal present
        
    if background_input: # Background signal present
        background_hists = []
        # Make histograms
        for back_data, back_weight, label in zip(background_x, background_weights, background_labels):
            # hist.storage.Weight() for MC data
            back_hist = Hist.new.Reg(num_bins, xmin, xmax, name=label).Weight()
            back_hist.fill(back_data, weight=back_weight)
            # For text annotations
            text.append(f'({bullets}) {label}: Weighted Sum (value = {back_hist.sum().value:.3e}, '
                        f'variance = {back_hist.sum().variance:.3e}),')
            text.append(f'Underflow = {back_hist.view(flow=True)[0].value:.3e}, Overflow = {back_hist.view(flow=True)[-1].value:.3e}')
            bullets += 1
            
            background_hists.append(back_hist)
            hists.append(back_hist)

        # Total count of background data and variance
        back_stacked_counts = sum(h.view(flow=False).value for h in background_hists) # This will be the starting bottom arg for stacking Signal histogram bars
        variances = sum(h.view(flow=False).variance for h in background_hists)
        stat_err = np.sqrt(variances) # Statistical uncertainty

        # Plot the MC bars
        # bottom for each histogram, will be updated in the loop to stack histograms
        bottom = np.zeros_like(back_stacked_counts)
        for h, color, label in zip(background_hists, background_colors, background_labels):
            # Plot stacked histogram bars
            counts = h.view(flow=False).value
            main_axes.bar(x=bin_centres, height=counts, width=widths, bottom=bottom, color=color, label=label, align='center')
            bottom += counts # To stack histograms

        if show_back_unc:
            # Plot the statistical uncertainty
            main_axes.bar(x=bin_centres, height=stat_err*2, width=widths, bottom=bottom-stat_err, color='none', hatch="////", label='Stat. Unc.', align='center')
        
    else: # No background MC present, Signal histogram will be stacked at y=0
        back_stacked_counts = np.zeros(len(bin_centres))
    
    if signal_input: # MC signal present
        signal_hists = []
        # Make histograms
        for signal_data, signal_weight, label in zip(signal_x, signal_weights, signal_labels):
             # hist.storage.Weight() for MC data
            signal_hist = Hist.new.Reg(num_bins, xmin, xmax, name=label).Weight()
            signal_hist.fill(signal_data, weight=signal_weight)
            # For text annotations
            text.append(f'({bullets}) {label}: Weighted Sum (value = {signal_hist.sum().value:.3e}, '
                        f'Variance = {signal_hist.sum().variance:.3e}),')
            text.append(f'Underflow = {signal_hist.view(flow=True)[0].value:.3e}, '
                        f'Overflow = {signal_hist.view(flow=True)[-1].value:.3e}')
            bullets += 1
            
            signal_hists.append(signal_hist)
            hists.append(signal_hist)
            
        # Total count of signal data
        stacked_counts = sum(h.view(flow=False).value for h in signal_hists)

        # Plot the bars
        # bottom starts from the y-values of stacked background histogram bin values
        bottom = back_stacked_counts.copy() # Will be updated in the loop to stack Signal histogram
        for h, color, label in zip(signal_hists, signal_colors, signal_labels):
            counts = h.view(flow=False).value
            main_axes.bar(x=bin_centres, height=counts, width=widths, bottom=bottom, color=color,
                    label=label, align='center')
            bottom += counts
    else: # No Signal present, no signal stacked
        stacked_counts = np.zeros(len(bin_centres))

    mc_total = stacked_counts + back_stacked_counts

    # Plot residual plot
    if residual_axes is not None:
        if not np.all(mc_total == 0) and not np.all(data_x == 0):
            # won't raise warnings for division by zero and invalid operation
            with np.errstate(divide='ignore', invalid='ignore'):
                # Get ratio when mc_total is not zero otherwise return nan
                ratio = np.divide(data_x, mc_total, out=np.full_like(data_x, np.nan), where=mc_total != 0)
                # Get error when data bin values are not zero otherwise return zero
                yerr = np.divide(np.abs(ratio * data_x_errors),
                                 data_x,
                                 out=np.zeros_like(data_x),  # set to 0 when undefined
                                 where=data_x != 0)
            residual_axes.errorbar(bin_centres, ratio, yerr=yerr, fmt='ko')
            residual_axes.axhline(1, color='r', linestyle='--')
            residual_axes.set_xlabel(x_label, fontsize=label_fontsize,
                                     x=1, horizontalalignment='right' )
            residual_axes.set_ylabel('Ratio (Data/MC)', fontsize=label_fontsize,
                                     y=1, horizontalalignment='right')
            residual_axes.xaxis.set_minor_locator(AutoMinorLocator())
            residual_axes.yaxis.set_minor_locator(AutoMinorLocator())
            # set the axis tick parameters for the main axes
            residual_axes.tick_params(which='both', # ticks on both x and y axes
                                      direction='in', # Put ticks inside and outside the axes
                                      labelsize=tick_labelsize, # Label size
                                      top=True, # draw ticks on the top axis
                                      right=True) # draw ticks on right axis
            if residual_plot_ylim is not None: # Set y limit
                residual_axes.set_ylim(residual_plot_ylim[0], residual_plot_ylim[1])
    return bin_centres, hists, text

# Helper function to plot_stacked_hist to validate input
def validate_plotting_input(data_dict, color_list, num_bins, xmin, xmax, fig_size,
                            ylim, residual_plot_ylim):
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dict.')

    if not isinstance(color_list, (list, tuple)):
        raise TypeError('Input for colors must be a list.')

    # Check if sample keys and color list have the same length
    if len(data_dict) != len(color_list):
        raise ValueError("Input lists for sample keys and colors must have the same length.") 
        raise ValueError("Sample keys and color list must match in length. "
            f"Got {len(data_dict)} and {len(color_list)}")

    # Validate num_bins
    if not isinstance(num_bins, int):
        raise TypeError(f'num_bins must be an int. Got {num_bins}')
    if not num_bins > 1:
        raise ValueError(f'num_bins must be greater than one! Got "{num_bins}"')
        
    # Validate xmin and xmax
    if not isinstance(xmin, (int, float)):
        raise TypeError(f'x_min must be a number. Got "{x_min}"')
    if not isinstance(xmax, (int, float)):
        raise TypeError(f'x_max must be a number. Got "{x_max}"')
    if xmax <= xmin:
        raise ValueError(f'x_max must be greater than x_min. Got x_max = "{x_max}", x_min = "{x_min}"')

    # Validate fig_size
    if (not isinstance(fig_size, tuple)
        or not all(isinstance(value, (int, float)) for value in fig_size)):
        raise ValueError("fig_size must be a tuple of two numbers.")
    if len(fig_size) != 2 or not all(value > 0 for value in fig_size):
        raise ValueError("fig_size must be a tuple of two positive numbers.")

    # Validate ylim
    if ylim is not None:
        if (not isinstance(ylim, tuple)
            or not all(isinstance(value, (int, float)) for value in ylim)):
            raise ValueError("ylim must be a tuple of two numbers.")
        if ylim[1] <= ylim[0]:
            raise ValueError(f"ylim[0] must be smaller than ylim[1]. Got ylim[0] '{ylim[0]}' and ylim[1] '{ylim[1]}'.")

    # Validate ylim for residual plot
    if residual_plot_ylim is not None:
        if (not isinstance(residual_plot_ylim, tuple)
            or not all(isinstance(value, (int, float)) for value in residual_plot_ylim)):
            raise ValueError("residual_plot_ylim must be a tuple of two numbers.")
        if residual_plot_ylim[1] <= residual_plot_ylim[0]:
            raise ValueError(f"residual_plot_ylim[0] must be smaller than residual_plot_ylim[1]. Got residual_plot_ylim[0] '{residual_plot_ylim[0]}' and residual_plot_ylim[1] '{residual_plot_ylim[1]}'.")


# This function plots one variable in one figure and calls stacked_histogram
# Example, data_dict = {
# 'Data' : Ak.array(...),
# 'Signal' : Ak.array(...),
# 'Background' : Ak.array(...)
# }
def plot_stacked_hist(data_dict, # see above
                      plot_variable, # Variable to plot
                      color_list, # color for each key in data_dict
                      num_bins, # Number of bins
                      xmin, # Left edge of first bin
                      xmax, # Right edge of last bin
                      x_label, # x-axis label
                        # Optional arg
                        y_label=None, # y axis label
                        ylim=None, # tuple of 2 numbers of y-axis limit
                        fit=None, # an array to overlay on the plot 
                        fit_label='fit', # legend label for fit
                        fit_fmt='-r', # plt.error format for fit
                        logy=False, # Whether to set log y axis
                        title=None, # Str for title
                        marker='o', # Marker type
                        title_fontsize=17, # Fontsize for title
                        label_fontsize=17, # Fontsize for x and y axes
                        legend_fontsize=17, # Fontsize for legend
                        tick_labelsize=15, # Fontsize for x and y axes ticks
                        text_fontsize=14, # Fontsize for text that shows histogram info
                        fig_size=(12, 8), # Figure size
                        show_text=False, # Whether to show the text that displays histogram info
                        show_back_unc=True, # Whether to show the background uncertainty
                        save_fig=False, # Whether to save figure
                        fig_name=None, # Filename of the image. If not provided, save figure using the plot_variable and the keys of data_dict
                        residual_plot=False, # Whether to plot residual plot
                        residual_plot_ylim=None # A tuple of 2 numbers. Residual plot y-axis limit
                   ):

    # Validate input
    validate_plotting_input(data_dict, color_list, num_bins, xmin, xmax, fig_size, ylim, residual_plot_ylim)

    time_start = time.time()

    if residual_plot:
        # Create main plot and residual subplot
        fig, (main_axes, residual_axes) = plt.subplots(2, 1, figsize=fig_size, 
                                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                                       sharex=True)
    else:
        fig, main_axes = plt.subplots(figsize=fig_size)
        residual_axes = None

    # Plot stacked histograms
    bin_centres, hists, text = stacked_histogram(data_dict, color_list, plot_variable, xmin, xmax, num_bins, main_axes, marker, show_back_unc, residual_axes, x_label, label_fontsize, tick_labelsize, residual_plot_ylim)

    # Plot fit
    if fit is not None:
        if len(fit) != len(bin_centres):
            raise ValueError('The array for fitted data must have the same length as the bin centres. Perhaps you used the wrong num_bins, xmin, or xmax?')
        main_axes.plot(bin_centres, fit, fit_fmt, label=fit_label)

    # Text annotations
    if show_text:
        if residual_plot:
            for i, line in enumerate(text):
                residual_axes.text(-0.05, -0.2 - i * 0.15, line, ha='left', va='top', transform=residual_axes.transAxes, fontsize=text_fontsize)
        else:
            for i, line in enumerate(text):
                main_axes.text(-0.05, -0.2 - i * 0.05, line, ha='left', va='top', transform=main_axes.transAxes, fontsize=text_fontsize)
    # Separation of x axis minor ticks for main axes
    main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
    
    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          labelsize=tick_labelsize, # Label size
                          top=True, # draw ticks on the top axis
                          right=True) # draw ticks on right axis

    if not residual_plot: # else: set x_label with residual axes and x_label shared with main axes
        # x-axis label
        main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                             x=1, horizontalalignment='right' )

    # set y-axis label for main axes
    if not y_label:
        y_label = 'Events'
    main_axes.set_ylabel(y_label,
                         fontsize=label_fontsize,
                         y=1, horizontalalignment='right') 
    
    # Set legend and title
    main_axes.legend(frameon=False, fontsize=legend_fontsize) 
    main_axes.set_title(title, fontsize=title_fontsize)

    if ylim is not None: # Set y-axis limit for main axes
        main_axes.set_ylim(ylim[0], ylim[1])
    
    if logy: # y-axis log scale
        main_axes.set_yscale('log')
        if residual_plot:
            residual_axes.set_yscale('log')

    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    plt.tight_layout()
    plt.show()

    # Save figure
    if save_fig:
        if not fig_name:
            fig_name = f'{plot_variable}_'
            for i in data_dict.keys():
                fig_name += i
        fig.savefig(str(fig_name), dpi=500)

    return fig, hists
# End of plot_stacked_hist() function    

# Plot 2D histogram
def histogram_2d(data, # A tuple/list of two arrays for histogram along x and y axis
                 num_bins, # A tuple/list of 2 numbers corresponding to the number of bins for
                           # histogram along x and y axis
                 min_max, # A tuple/list of 2 tuples, each with 2 numbers corresponding to the
                          # bin range for histogram along x and y axis
                 label, # A tuple/list of str
                 label_fontsize=12, 
                 tick_labelsize=10,
                 title_fontsize=13, 
                 title='', 
                 colorbar_label='' # Label for colorbar
                ):
    # Validate variable
    if (not isinstance(data, (list, tuple)) or
        not all(isinstance(i, ak.Array) for i in data)
       ):
        raise TypeError("data must be a list or tuple of two awkward arrays.")
        
    # Validate the length of input
    if (
        len(data) != 2
        or len(num_bins) != 2 or len(min_max) != 2
        or len(label) != 2
        ):
        raise ValueError('Each input must have exactly two elements.')

    # Validate the number of bins
    if not all(isinstance(i, int) for i in num_bins): # Must be an int
        raise TypeError('num_bins have to be a tuple or list of numbers.')
    if not all(i > 0 for i in num_bins): # Must be positive
        raise ValueError('num_bins have to be a tuple or list of two positive numbers.')

    # Validate the min and max points for x and y
    for pair in min_max:
        if len(pair) != 2:
            raise ValueError(f'Expect a tuple of two numbers for min_max. Got {pair}')
        if not all(isinstance(i, (int, float)) for i in pair):
            raise TypeError('min_max must be a pair of tuple/list of two numbers.')
        if not (pair[1] - pair[0]) > 0:
            raise ValueError('The second number in each tuple/list of min_max must be larger than the first.')

    # Label of axes
    if isinstance(label, str):
        raise TypeError(f'label must be a list or tuple of two str. Got a str instead.')
    # Convert tuple input into list for assignment
    if isinstance(label, tuple):
        label = [label[0], label[1]]
    # Assign variable to label if label for x or/and y is an empty string
    for i, j in enumerate(label):
        if j:
            label[i] = str(j)
        else:
            label[i] = variable[i]

    # Extract values for x and y from inputs
    data_x, data_y = data
    num_bins_x, num_bins_y = num_bins
    (xmin, xmax), (ymin, ymax) = min_max
    label_x, label_y = label

    # 2D Histogram
    h = Hist(
        hist.axis.Regular(num_bins_x, xmin, xmax, name=label_x, label=label_x, flow=False),
        hist.axis.Regular(num_bins_y, ymin, ymax, name=label_y, label=label_y, flow=False)
    )
    h.fill(ak.to_numpy(data_x), ak.to_numpy(data_y))

    # Plot 2D histogram
    fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
    values, x_bin_edges, y_bin_edges = h.to_numpy()
    mesh = ax.pcolormesh(x_bin_edges, y_bin_edges, values.T, cmap="viridis")
    # Set axes label, ticks, and title
    ax.set_xlabel(label_x, fontsize=label_fontsize)
    ax.set_ylabel(label_y, fontsize=label_fontsize)
    ax.tick_params(which='both', labelsize=tick_labelsize)
    ax.set_title(str(title), fontsize=title_fontsize)
    
    # Create colorbar and adjust its label and tick label sizes
    cbar = fig.colorbar(mesh)
    cbar.set_label(colorbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    plt.show()
    return fig, h

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Helper functions to plot_histograms()

# Validate x_label_list. If a str is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each label in the x_label_list corresponds to one variable in plot_variables
def validate_xlabel(x_label_list, plot_variables):
    variable_count = len(plot_variables)
    x_label = []
    if isinstance(x_label_list, str): # If input is a string
        x_label = [x_label_list] * variable_count
    elif isinstance(x_label_list, (list, tuple)):
        if len(x_label_list) == 0: # If input is an empty list
            for variable in plot_variables:
                x_label.append(variable)
        elif len(x_label_list) == variable_count:
            for i in x_label_list:
                x_label.append(str(i))
        else:
            raise ValueError("Input list/tuple for x axis label must have "
                             " the same length as variable list.")
    else:
         raise TypeError("Input for x axis label must be a str or a list/tuple with"
                         " the same length as variable list.")
    return x_label

# Validate y-axis_list. If a str is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each label in the y_label_list corresponds to one variable in plot_variables
def validate_ylabel(y_label_list, variable_count):
    if y_label_list is None:
        y_label_list = 'Events'
    y_label = []
    if isinstance(y_label_list, str): # If input is a string
        y_label = [y_label_list] * variable_count
    elif isinstance(y_label_list, (list, tuple)):
        if len(y_label_list) == 0: # If input is empty
            for i in range(variable_count):
                y_label.append('')
        elif len(y_label_list) == variable_count:
            for i in y_label_list:
                y_label.append(str(i))    
        else:
            raise ValueError("Input list/tuple for y axis label must have"
                             " the same length as variable list.")
    else:
        raise TypeError("Input for y axis label must be a str or a list/tuple with"
                        " the same length as variable list.")
    return y_label

# Validate title_list. If a str is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each entry in the title_list corresponds to one variable in plot_variables
def validate_title(title_list, variable_count):
    if title_list is None:
        title_list = ''
    title = []
    if isinstance(title_list, str): # If input is a string
        title = [title_list] * variable_count
    elif isinstance(title_list, (list, tuple)):
        if len(title_list) == 0: # If input is empty
            for i in range(variable_count):
                title.append('')
        elif len(title_list) == variable_count:
            for i in title_list:
                title.append(str(i))    
        else:
            raise ValueError("Input list/tuple for title must have"
                             " the same length as variable list.")
    else:
        raise TypeError("Input for title must be a str or a list/tuple with"
                        " the same length as variable list.")
    return title

# Validate num_bins_list. If an int is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each entry in the num_bins_list corresponds to one variable in plot_variables
def validate_num_bins(num_bins_list, variable_count):
    num_bins = []
    
    if isinstance(num_bins_list, int):
        num_bins = [num_bins_list] * variable_count
    elif isinstance(num_bins_list, (list, tuple)):
        if len(num_bins_list) == variable_count:
            try:
                for i in num_bins_list:
                    num_bins.append(int(i))
            except TypeError:
                print("Input for number of bins must be an int or"
                      " a list/tuple of int with the same length as variable "
                      "list.")
                raise
            except Exception as e:
                print(f"Unexpected exception : {e}")
                raise
        else:
            raise ValueError("Invalid list/tuple length for num_bins_list.")
    else: # Invalid input type
        raise TypeError("Input for number of bins must be an int or a"
                        " list/tuple of int with the same length as variable list.")
    return num_bins

# Validate xmin_xmax_list. If a tuple of 2 numbers is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each tuple of 2 numbers in the xmin_xmax_list corresponds to one variable in plot_variables
def validate_xmin_xmax(xmin_xmax_list, variable_count):
    if isinstance(xmin_xmax_list, str):
        raise TypeError("Input for xmin and xmax must not be a str.")

    xmin_xmax = [] # Store validated input

    # Case 1: Single (xmin, xmax) tuple/list for all variables
    if (isinstance(xmin_xmax_list, (list, tuple)) # Only accept a tuple or list, avoid str input
        and len(xmin_xmax_list) == 2 # Only accept (a, b) or [a, b]
        # where a and b are int or float
        and all(isinstance(value, (int, float)) for value in xmin_xmax_list)
       ): 
        xmin, xmax = xmin_xmax_list
        # Raise error if xmax is smaller than or equal to xmin
        if xmax <= xmin:
            raise ValueError(f"Input xmin = {xmin}, xmax = {xmax} : xmax must be greater than xmin.")
        # Make xmin_xmax a list with the same length as variable list
        xmin_xmax = [(xmin, xmax)] * variable_count
        return xmin_xmax

    # Case 2: List of (xmin, xmax) pairs for each variable.
    if (isinstance(xmin_xmax_list, (list, tuple)) # Avoid str input
        and len(xmin_xmax_list) == variable_count # Number of pairs must match number of variables
       ):
        for pair in xmin_xmax_list:
            # Raise error if each object in xmin_xmax_list is not a tuple or list of 2 numbers
            if not isinstance(pair, (list, tuple)):
                raise TypeError("Each element must be a tuple/list of two numbers (xmin, xmax).")

            if len(pair) != 2:
                raise ValueError("Each element must be a tuple/list of two numbers (xmin, xmax).")
                
            xmin, xmax = pair
            # Only accept (a, b) or [a, b] where a and b are int or float
            if not all(isinstance(value, (int, float)) for value in (xmin, xmax)):
                raise TypeError(f"Input xmin = {xmin}, xmax = {xmax} : xmin and xmax must be int or float.")
            # Raise error if xmax is smaller than or equal to xmin
            if xmax <= xmin:
                raise ValueError(f"Input xmin = {xmin}, xmax = {xmax} : xmax must be greater than xmin.")
            # Update with validated input    
            xmin_xmax.append((xmin, xmax))
        return xmin_xmax

    # If none of the above formats match
    raise ValueError(
        "Invalid format for xmin_xmax. Must be either:\n"
        "1. A tuple/list of two numbers: (xmin, xmax)\n"
        "2. A list of (xmin, xmax) tuples/lists of same length as variable count. "
        f"Number of input variables = {variable_count}."
    )  
            

# Validate ylim_list. If a tuple of 2 numbers is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each tuple of 2 numbers in the ylim_list corresponds to one variable in plot_variables
def validate_ylim_list(ylim_list, variable_count):
    if ylim_list is None:
        return [None] * variable_count
    if isinstance(ylim_list, str):
        raise TypeError("Input for ylim_list must not be a str.")

    ylim = [] # Store validated input

    # Case 1: Single ylim tuple/list for all variables
    if (isinstance(ylim_list, (list, tuple)) # Only accept a tuple or list, avoid str input
        and len(ylim_list) == 2 # Only accept (a, b) or [a, b]
        # where a and b are int or float
        and all(isinstance(value, (int, float)) for value in ylim_list)
       ): 
        ymin, ymax = ylim_list
        # Raise error if ymax is smaller than or equal to ymin
        if ymax <= ymin:
            raise ValueError(f"Input ylim_list = {ylim_list} : ylim_list[1] must be greater than ylim_list[0].")
        # Make ylim a list with the same length as variable list
        ylim = [(ymin, ymax)] * variable_count
        return ylim

    # Case 2: List of ylim_list for each variable.
    if (isinstance(ylim_list, (list, tuple)) # Avoid str input
        and len(ylim_list) == variable_count # Number of pairs must match number of variables
       ):
        for pair in ylim_list:
            # Raise error if each object in ylim_list is not a tuple or list of 2 numbers
            if not isinstance(pair, (list, tuple)):
                raise TypeError(f"Each element of ylim_list must be a tuple/list of two numbers. Got {pair}")

            if len(pair) != 2:
                raise ValueError(f"Each element of ylim_list must be a tuple/list of two numbers. Got {pair}")
                
            ymin, ymax = pair
            # Only accept (a, b) or [a, b] where a and b are int or float
            if not all(isinstance(value, (int, float)) for value in (ymin, ymax)):
                raise TypeError(f"Input in ylim_list = {pair} : Both numbers must be int or float.")
            # Raise error if ymax is smaller than or equal to ymin
            if ymax <= ymin:
                raise ValueError(f"Input in ylim_list = {pair} : The second number must be greater than the first.")
            # Update with validated input    
            ylim.append((ymin, ymax))
        return ylim

    # If none of the above formats match
    raise ValueError(
        "Invalid format for ylim_list. Must be either:\n"
        "1. A tuple/list of two numbers\n"
        "2. A list of (ymin, ymax) tuples/lists of same length as variable count. "
        f"Number of input variables = {variable_count}."
    )  


# Validate residual_ylim_list. If a tuple of 2 numbers is provided, apply that to all plots
# If a list is given, it has to have same length as plot_variables list
# Each tuple of 2 numbers in the residual_ylim_list corresponds to one variable in plot_variables
def validate_residual_ylim_list(residual_ylim_list, variable_count):
    if residual_ylim_list is None:
        return [None] * variable_count
    if isinstance(residual_ylim_list, str):
        raise TypeError("Input for residual_ylim_list must not be a str.")

    residual_ylim = [] # Store validated input

    # Case 1: Single residual_ylim tuple/list for all variables
    if (isinstance(residual_ylim_list, (list, tuple)) # Only accept a tuple or list, avoid str input
        and len(residual_ylim_list) == 2 # Only accept (a, b) or [a, b]
        # where a and b are int or float
        and all(isinstance(value, (int, float)) for value in residual_ylim_list)
       ): 
        ymin, ymax = residual_ylim_list
        # Raise error if ymax is smaller than or equal to ymin
        if ymax <= ymin:
            raise ValueError(f"Input residual_ylim_list = {residual_ylim_list} : residual_ylim_list[1] must be greater than residual_ylim_list[0].")
        # Make residual_ylim a list with the same length as variable list
        residual_ylim = [(ymin, ymax)] * variable_count
        return residual_ylim

    # Case 2: List of residual_ylim_list for each variable.
    if (isinstance(residual_ylim_list, (list, tuple)) # Avoid str input
        and len(residual_ylim_list) == variable_count # Number of pairs must match number of variables
       ):
        for pair in residual_ylim_list:
            # Raise error if each object in residual_ylim_list is not a tuple or list of 2 numbers
            if not isinstance(pair, (list, tuple)):
                raise TypeError(f"Each element of residual_ylim_list must be a tuple/list of two numbers. Got {pair}")

            if len(pair) != 2:
                raise ValueError(f"Each element of residual_ylim_list must be a tuple/list of two numbers. Got {pair}")
                
            ymin, ymax = pair
            # Only accept (a, b) or [a, b] where a and b are int or float
            if not all(isinstance(value, (int, float)) for value in (ymin, ymax)):
                raise TypeError(f"Input in residual_ylim_list = {pair} : Both numbers must be int or float.")
            # Raise error if ymax is smaller than or equal to ymin
            if ymax <= ymin:
                raise ValueError(f"Input  in residual_ylim_list = {pair} : The second number must be greater than the first.")
            # Update with validated input    
            residual_ylim.append((ymin, ymax))
        return residual_ylim

    # If none of the above formats match
    raise ValueError(
        "Invalid format for residual_ylim_list. Must be either:\n"
        "1. A tuple/list of two numbers\n"
        "2. A list of (ymin, ymax) tuples/lists of same length as variable count. "
        f"Number of input variables = {variable_count}."
    )  


# This function aims to plot one or multiple variables in separate plots
# Example, data_dict = {
# 'Data' : Ak.array(...),
# 'Signal' : Ak.array(...),
# 'Background' : Ak.array(...)
# }
def plot_histograms(
        data_dict,
        plot_variables, # List of variables to plot
        color_list, # List of color (one for each key in data_dict)
        xmin_xmax_list, # Tuple of 2 numbers or list of tuples for bin range or x axis limit
        num_bins_list, # int or list of int for number of bins
        x_label_list, # str or list of str for x axis label
        # Optional arguments start from here
        y_label_list=None, # Str or list of str for y axis label
        ylim_list=None, # Tuple of 2 numbers or list of tuples for y axis limit
        logy=False, # Whether to set the y axis as log scale
        title_list=None, # Str or list of str for title
        marker='o', # Marker type
        title_fontsize=17, # Fontsize for title
        label_fontsize=17, # Fontsize for x and y axes
        legend_fontsize=17, # Fontsize for legend
        tick_labelsize=15, # Fontsize for x and y axes ticks
        text_fontsize=14, # Fontsize for text that shows histogram info
        fig_size=(12, 8), # Figure size
        show_text=False, # Bool - whether to show the text that displays histogram info
        show_back_unc=True, # Bool - whether to show the background uncertainty
        save_fig=False, # Whether to save figure
        residual_plot=False,
        residual_ylim_list=None # Tuple of 2 numbers or list of tuples for residual plot y-axis limit
    ):
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dict.')

    if isinstance(color_list, str):
        raise TypeError('Input for colors must be a list.')

    # Check if all input lists are the same length
    if len(data_dict) != len(color_list):
        raise ValueError("Input lists for sample keys and colors must have the same length.") 
        raise ValueError("Sample keys and color list must match in length. "
            f"Got {len(data_dict)} and {len(color_list)}")

    # Validate fig_size
    if (not isinstance(fig_size, tuple)
        or not all(isinstance(value, (int, float)) for value in fig_size)):
        raise ValueError("fig_size must be a tuple of two numbers.")
    if len(fig_size) != 2 or not all(value > 0 for value in fig_size):
        raise ValueError("fig_size must be a tuple of two positive numbers.")

    time_start = time.time()

    # Validate plot_variables
    if isinstance(plot_variables, (list, tuple)):
        variable_count = len(plot_variables)
    else:
        variable_count = 1 # It is a str
        plot_variables = [plot_variables] * variable_count

    # Validate x label input
    # If x_label_list is one string, make it into an array corresponding to the length of other input list
    x_label_list = validate_xlabel(x_label_list, plot_variables)

    # Similarly, validate ylabel input, title
    y_label_list = validate_ylabel(y_label_list, variable_count)
    title_list = validate_title(title_list, variable_count)
    
    # Validate number of bins input
    num_bins_list = validate_num_bins(num_bins_list, variable_count)
    # Validate xmin xmax input                
    xmin_xmax_list = validate_xmin_xmax(xmin_xmax_list, variable_count)
    # Validate main plot and residual plot ylim
    ylim_list = validate_ylim_list(ylim_list, variable_count)
    residual_ylim_list = validate_residual_ylim_list(residual_ylim_list, variable_count)

    fig_list = [] # Hold Figure, one entry per variable
    hists_list = [] # Hold list of Hist per variable

    # Plot variable one by one
    for (variable, x_label, y_label,
         xmin_xmax, num_bins, title,
         ylim, residual_ylim) in zip(plot_variables, x_label_list,
                                     y_label_list, xmin_xmax_list,
                                     num_bins_list, title_list, ylim_list, 
                                     residual_ylim_list):
        
        if residual_plot:
        # Create main plot and residual subplot
            fig, (main_axes, residual_axes) = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, main_axes = plt.subplots(figsize=fig_size)
            residual_axes = None
        
        xmin, xmax = xmin_xmax
        _, hists, text = stacked_histogram(data_dict, color_list, variable, xmin, xmax, 
                                           num_bins, main_axes, marker, show_back_unc, 
                                           residual_axes, x_label, label_fontsize, 
                                           tick_labelsize, residual_ylim)

        # Text annotations
        if show_text:
            if residual_plot:
                for i, line in enumerate(text):
                    residual_axes.text(-0.05, -0.2 - i * 0.15, line, ha='left', va='top', transform=residual_axes.transAxes, fontsize=text_fontsize)
            else:
                for i, line in enumerate(text):
                    main_axes.text(-0.05, -0.2 - i * 0.05, line, ha='left', va='top', transform=main_axes.transAxes, fontsize=text_fontsize)

        if ylim is not None: # Set y-axis limit
            main_axes.set_ylim(ylim[0], ylim[1])
            
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                              direction='in', # Put ticks inside and outside the axes
                              labelsize=tick_labelsize, # Label size
                              top=True, # draw ticks on the top axis
                              right=True) # draw ticks on right axis
    
        if not residual_plot: # else: main axes will share x label with residual axes
            # x-axis label
            main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                                 x=1, horizontalalignment='right' )
                
        # set y-axis label for main axes
        main_axes.set_ylabel(y_label,
                             fontsize=label_fontsize,
                             y=1, horizontalalignment='right') 
        
        # set the legend and title
        main_axes.legend(frameon=False, fontsize=legend_fontsize) # no box around the legend    
        main_axes.set_title(title, fontsize=title_fontsize)
        
        if logy:
            main_axes.set_yscale('log')

        if ylim:
            main_axes.set_ylim(ylim[0], ylim[1])
    
        elapsed_time = time.time() - time_start 
        print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

        # Show the plot
        plt.tight_layout()
        plt.show()

        
        # Save figure
        if save_fig:
            fig_name = f'{variable}_'
            for i in data_dict.keys():
                fig_name += i
            print(fig_name)
            fig.savefig(str(fig_name), dpi=500)

        
        hists_list.append(hists)
        fig_list.append(fig)

    return fig_list, hists_list
    