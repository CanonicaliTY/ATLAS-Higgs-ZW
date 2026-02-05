# string codes and their corresponding filepath to pre-written parquet files (produced using analysis_uproot())
directory = '../../parquet'
PARQUET_DICT = {
    '2to4lep' : f'{directory}/2to4lep',
    
    'Zee' : f'{directory}/Zee',
    'Zmumu' : f'{directory}/Zmumu',
    'Ztautau' : f'{directory}/Ztautau',
    'VBF_Zee' : f'{directory}/VBF_Zee',
    'VBF_Zmumu' : f'{directory}/VBF_Zmumu',
    'VBF_Ztautau' : f'{directory}/VBF_Ztautau',

    'Zee_BFil' : f'{directory}/Zee_BFil',
    'Zee_CFilBVeto' : f'{directory}/Zee_CFilBVeto',
    'Zee_CVetoBVeto' : f'{directory}/Zee_CVetoBVeto',
    'Zmumu_BFil' : f'{directory}/Zmumu_BFil',
    'Zmumu_CFilBVeto' : f'{directory}/Zmumu_CFilBVeto',
    'Zmumu_CVetoBVeto' : f'{directory}/Zmumu_CVetoBVeto',
    
    'Wenu' : f'{directory}/Wenu',
    'Wmunu' : f'{directory}/Wmunu',
    'Wtaunu' : f'{directory}/Wtaunu',
    'VBF_Wenu' : f'{directory}/VBF_Wenu',
    'VBF_Wmunu' : f'{directory}/VBF_Wmunu',
    'VBF_Wtaunu' : f'{directory}/VBF_Wtaunu',
    
    'ttbar' : f'{directory}/ttbar',
    'VV4l' : f'{directory}/VV4l',
    
    'm10_40_Zee' : f'{directory}/m10_40_Zee',
    'm10_40_Zmumu' : f'{directory}/m10_40_Zmumu',
    
    'ggH_H4l' : f'{directory}/ggH_H4l',
    'VBF_H4l' : f'{directory}/VBF_H4l',
    'WpH_H4l' : f'{directory}/WpH_H4l',
    'WmH_H4l' : f'{directory}/WmH_H4l',
    'ZH_H4l' : f'{directory}/ZH_H4l',
    'ggZH_H4l' : f'{directory}/ggZH_H4l',
    'ttH_H4l' : f'{directory}/ttH_H4l',

    'GamGam' : f'{directory}/GamGam',
    #'Hyy' : f'{directory}/Hyy',
    
    'ggF_Hyy' : f'{directory}/ggF_Hyy',
    'VBF_Hyy' : f'{directory}/VBF_Hyy',
    'WpH_Hyy' : f'{directory}/WpH_Hyy',
    'WmH_Hyy' : f'{directory}/WmH_Hyy',
    'ZH_Hyy' : f'{directory}/ZH_Hyy',
    'ggZH_Hyy' : f'{directory}/ggZH_Hyy',
    'ttH_Hyy' : f'{directory}/ttH_Hyy',
}

STR_CODE_COMBO = {
    'VBF_Zll' : 'VBF_Zee + VBF_Zmumu + VBF_Ztautau',
    'm10_40_Zll' : 'm10_40_Zee + m10_40_Zmumu',
    'Wlepnu' : 'Wenu + Wmunu + Wtaunu',
    'VBF_Wlepnu' : 'VBF_Wenu + VBF_Wmunu + VBF_Wtaunu',
    'H4l' : 'ggH_H4l + VBF_H4l + WpH_H4l + WmH_H4l + ZH_H4l + ggZH_H4l + ttH_H4l',
    'Hyy' : 'ggF_Hyy + VBF_Hyy + WpH_Hyy + WmH_Hyy + ZH_Hyy + ggZH_Hyy + ttH_Hyy'
}

# Valid string codes
VALID_STR_CODE = list(PARQUET_DICT.keys()) + list(STR_CODE_COMBO.keys())
