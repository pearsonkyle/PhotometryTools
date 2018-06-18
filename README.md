# PhotometryTools
Tools for performing aperture and PSF photometry for real time data analysis

## Features
- Centroid Finding (2D Gaussian Optimization)
- Aperature Photometry (Fractional pixel values handled by interpolating onto higher resolution grid) 
- PSF Photometry (just use best fit Gaussian parameters from centroid algorithm)
- Sky Background Subtraction (can handle fractional pixels within an annulus)
- Data plotter (excellent for analyzing data in real time while taking observations)
- Data Generator 

### Requirements
- Python 3
- Numpy
- Matplotlib
- Scipy
- Astropy

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


## Centroid Finding
create plot that has X/Y collapsed profiles and imshow with dot 
