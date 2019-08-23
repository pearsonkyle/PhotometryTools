import re
import requests
import numpy as np

def get_ld(priors, band='Spit36'):

    url = 'http://astroutils.astronomy.ohio-state.edu/exofast/quadld.php'
    
    form = {
        'action':url,
        'pname':'Select Planet',
        'bname':band,
        'teff':priors['T*'],
        'feh':priors['FEH*'],
        'logg':priors['LOGG*']
    }
    session = requests.Session()
    res = session.post(url,data=form)
    lin,quad = re.findall("\d+\.\d+",res.text)
    return float(lin), float(quad)

if __name__ == "__main__":
    priors = {
        'T*':6000,
        'FEH*':0.25,
        'LOGG*':4.0,
    }

    lin,quad = get_ld(priors,'Spit36')