# PhotometryTools
Tools for performing aperture and PSF photometry for real time data analysis

## Features
- Centroid Finding (2D Gaussian Optimization)
- Aperature Photometry (Fractional pixel values handled by interpolating onto higher resolution grid) 
- PSF Photometry (just use best fit Gaussian parameters from centroid algorithm)
- Sky Background Subtraction (can handle fractional pixels within an annulus)
- Data Generator 

### Requirements
- Python 3.4+
- Numpy
- Matplotlib
- Scipy


### Example Code
```python
from PhotometryTools import ccd, psf, fit_centroid, phot

if __name__ == "__main__":

    # generate some test data
    img = ccd([1024,1024])
    star = psf(256,512,2000,4,4,0,0)
    img.draw(star)

    # fit 2D Gaussian to a location near [250,506] as initial guess
    pars_psf = fit_centroid(img.data,[250,506],box=25,psf_output=False)
    
    # Sum up data using a circular aperture of 15 pixels around the centroid
    area = phot(pars_psf[0],pars_psf[1],img.data,r=15,debug=False,bgsub=True)
    print('best fit parameters:',pars_psf)
    print('phot area=',area)
    print('psf area=',star.gaussian_area)
```


### Data Generator
```python 
img = ccd([1024,1024]) # define a grid size or pass it a 2D np array
star = psf( np.random.normal(492,5), # x - centroid
             np.random.normal(751,5), # y - centroid
             np.random.normal(2000,100), # flux 
             np.random.normal(4,0.2), # x - standard deviation
             np.random.normal(4,0.2), # y - standard deviation
             0, # background level 
             0 ) # rotation of PSF 
img.draw(star)
```

The above code should give you something like this: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

If you want to make a fits file use the data generated above and add: 
```python 
from astropy.io import fits 
from datetime import datetime

# save to fits file
header = fits.Header( {'TIME':datetime.now().strftime("%H:%M:%S.%f")} )
hdu = fits.PrimaryHDU(img.data,header=header)
hdul = fits.HDUList([hdu])
hdul.writeto("test.fits")
```

