# String codes and their dataset identifiers
DIDS_DICT = {
    # Z->ll
    'Zee' : [700322, 700321, 700320],
    'Zee_BFil' : [700320],
    'Zee_CFilBVeto' : [700321],
    'Zee_CVetoBVeto' : [700322],
    'Zmumu' : [700325, 700324, 700323],
    'Zmumu_BFil' : [700323],
    'Zmumu_CFilBVeto' : [700324],
    'Zmumu_CVetoBVeto' : [700325],
    'Ztautau' : [700792, 700793, 700794],
    # VBF Z->ll
    'VBF_Zee' : [700358], 'VBF_Zmumu' : [700359],
    'VBF_Ztautau' : [700360],
    # W->lv
    'Wenu' : [700338, 700339, 700340],
    'Wmunu' : [700341, 700342, 700343],
    'Wtaunu' : [700344, 700345, 700346, 700347, 700348, 700349],
    # VBF W->lv
    'VBF_Wenu' : [700362],
    'VBF_Wmunu' : [700363],
    'VBF_Wtaunu' : [700364],
    # ttbar
    'ttbar' : [410470],
    # ttbar + V
    'ttV' : [410218,410219,410155],
    # VV->llll
    'VV4l' : [700600, 700587, 700591],
    # VV->lllnu
    'VV3l' : [700601],
    # VVV->multileptons
    'VVV' : [364242, 364243, 364245, 364246, 364247, 364248],
    # ZZ->llbb
    'VVllbb' : [700494],
    # Low-mass Drell-Yan samples
    'm10_40_Zee' : [700467, 700468, 700469],
    'm10_40_Zmumu' : [700470, 700471, 700472],
    'm10_40_Ztautau' : [700901, 700902, 700903],
    # H->yy
    'ggF_Hyy' : [343981], 'VBF_Hyy' : [346214], 'WpH_Hyy' : [345318],
    'WmH_Hyy' : [345317], 'ZH_Hyy' : [345319], 'ggZH_Hyy' : [345061],
    'ttH_Hyy' : [346525],
    # H->llll
    'ggH_H4l' : [345060], 'VBF_H4l' : [346228],
    'ttH_H4l' : [346340, 346341, 346342],
    'ggZH_H4l' : [345066], 'ZH_H4l' : [346645],
    'WpH_H4l' : [346646], 'WmH_H4l' : [346647],
    }

# Create records with generic name
DIDS_DICT['VBF_Zll'] = (DIDS_DICT['VBF_Zee'] + 
                        DIDS_DICT['VBF_Zmumu'] + 
                        DIDS_DICT['VBF_Ztautau'])
DIDS_DICT['m10_40_Zll'] = (DIDS_DICT['m10_40_Zee'] + 
                           DIDS_DICT['m10_40_Zmumu'] + 
                           DIDS_DICT['m10_40_Ztautau'])
DIDS_DICT['Wlepnu'] = (DIDS_DICT['Wenu'] + 
                       DIDS_DICT['Wmunu'] + 
                       DIDS_DICT['Wtaunu'])
DIDS_DICT['VBF_Wlepnu'] = (DIDS_DICT['VBF_Wenu'] + 
                           DIDS_DICT['VBF_Wmunu'] + 
                           DIDS_DICT['VBF_Wtaunu'])
DIDS_DICT['Hyy'] = (DIDS_DICT['ggF_Hyy'] + 
                    DIDS_DICT['VBF_Hyy'] +
                    DIDS_DICT['WpH_Hyy'] +
                    DIDS_DICT['WmH_Hyy'] + 
                    DIDS_DICT['ZH_Hyy'] +
                    DIDS_DICT['ggZH_Hyy'] +
                    DIDS_DICT['ttH_Hyy'])
DIDS_DICT['H4l'] = (DIDS_DICT['ggH_H4l'] + 
                      DIDS_DICT['VBF_H4l'] +
                      DIDS_DICT['WpH_H4l'] +
                      DIDS_DICT['WmH_H4l'] + 
                      DIDS_DICT['ZH_H4l'] +
                      DIDS_DICT['ggZH_H4l'] +
                      DIDS_DICT['ttH_H4l'])
              
VALID_SKIMS = ['2to4lep', '2muons', 'GamGam', 'exactly4lep', '1LMET30', '3J1LMET30', 
              '2J2LMET30', '2bjets', '3lep', 'exactly3lep', '4lep']
