import numpy as np

def makeselection_ipta(excludepta=[]):
    """
    This is for setting white noise flags properly.

    For EFAC and EQUAD excludepta should be an empty list.

    For ECORR excludepta should be 'EPTA'.

    """
    def selection_ipta(psr):
        ptas = sorted(set(psr.flags['pta']))

        selection = np.full(len(psr.toas), fill_value='', dtype=object)
        for pta in ptas:
            if pta in excludepta:
                continue

            mask = psr.flags['pta'] == pta

            if pta in ['NANOGrav', 'MPTA']:
                selection[mask] = psr.flags['f'][mask]
            else:
                selection[mask] = psr.flags['group'][mask]

        return selection

    return selection_ipta

