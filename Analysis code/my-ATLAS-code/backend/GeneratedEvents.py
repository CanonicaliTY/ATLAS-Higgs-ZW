# Product of cross section, filter efficiency and k_factor for the various processes likely to be considered as signal for the purposes of efficiency calculations

CROSS_SECTION = {
    'Zee' :     2221.3,
    'Zmumu' :   2221.3,
    'Ztautau' : 2221.3,
    'Zll' :     2221.3,
    'ttbar' :   831.76,
    'Hyy' :     0.12516, 
    'H4l' :     0.007179
    }
 
def produced_event_count(dataKey,luminosity=36.6):
  
    # Function which returns the total number (sum of weights) of generated Monte Carlo events for the data set specified by 'dataKey' and the specificed integrated luminosity.
    # Return 0 for a real data set, or for an unrecognised data set.
  
    if dataKey in CROSS_SECTION.keys():
        xsec = CROSS_SECTION[dataKey]
        number_produced = xsec * luminosity * 1000
        print(f'The total number of produced events (sum of weights) for Monte Carlo data set {dataKey} and luminosity {luminosity} inverse femtobarns is {number_produced:.0f}')
        return number_produced
    elif dataKey in ['2to4lep', 'GamGam']:
        print('The total number of produced events does not have a useful meaning for a real ATLAS data set such as ',dataKey,'. Please discuss with your demonstrator if it is not clear why!', sep='')
    else:
        print(f'Dataset:  {dataKey}  is not recognised')
        print('The total numbers of produced events are currently provided only for the following data sets', list(CROSS_SECTION.keys()))
    return 0