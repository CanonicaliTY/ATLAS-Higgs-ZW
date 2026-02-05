import awkward as ak

# Relevant weight variables for different skims
WEIGHT_VAR = {
    '2to4lep' : ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP", 
                 "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER"],
    'exactly4lep' : ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP", 
                 "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER"],
    'GamGam' : ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP", "ScaleFactor_PHOTON"]
}

# Calculate the total weight for an event by multiplying all the important weights
def calculate_weight(events, luminosity, skim):
    if "sum_of_weights" not in events.fields:
        raise KeyError('Variable "sum_of_weights" was not found.')
    total_weight = luminosity * 1000 / events["sum_of_weights"] # * 1000 to go from fb-1 to pb-1
    
    for weight_var in WEIGHT_VAR[skim]:
        if weight_var not in events.fields:
            raise KeyError(f'Weight variable {weight_var} was not found.')
        total_weight = total_weight * events[weight_var]

    if ak.all(total_weight == 0): # Assume real data
        total_weight = ak.Array([1] * len(total_weight)) 
    return total_weight