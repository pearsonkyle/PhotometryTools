#python3.5+
from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from Photometry import fit_centroid, phot, psf

#from deinterlace import deinterlace

#DELACED
# DATE-OBS= '2018-10-27T23:14:15.016' / Date and time of observation

if __name__ == "__main__":
    dark = fits.open('hatp16_dark8000.fits')

    imgs = glob.glob('hatp16_[0-9]*.fits')

    sortedimgs = sorted(imgs, key=lambda name:int(name.split('_')[-1].split('.fits')[0]) )

    times = []; count = []
    xpos = []; ypos = []; sigma=[];
    bg = []; flux = [];

    x0 = 1110
    y0 = 956
    
    for i in range(len(sortedimgs)):

        img = fits.open( sortedimgs[i] ) 

        #if img[0].header.get('DELACED') != 'True':
        #    img = deinterlace( img )

        data = dark[0].data.astype(np.float32) - img[0].data.astype(np.float32)

        try:
            if i > 0:
                xc,yc,a,sigx,sigy,b = fit_centroid(data,[xpos[-1],ypos[-1]])
            else:
                xc,yc,a,sigx,sigy,b = fit_centroid(data,[x0,y0])    
        except:
            print('failed to fit centroid for:',sortedimgs[i])
            continue        

        area = phot(xc,yc,data,r=15,debug=False,bgsub=True)

        #t = img[0].header["UTC"].split(' ')[1]
        times.append( datetime.strptime(img[0].header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f") )
        xpos.append(xc)
        ypos.append(yc)
        sigma.append( 0.5*(sigx+sigy) )
        bg.append(b)
        flux.append(area)

        #completion = int( 100*i/len(imgs) )
        #if completion%2==0:
        #    print( "completion: {:.2f} %".format(100*i/len(imgs)) )


    fi = 0
    print('lowest count:',count[fi])
    lowest_time = times[fi]
    
    dt = [ (times[i]-lowest_time).seconds/60. for i in range(len(times)) ]

    f,ax = plt.subplots(4)
    
    ax[0].plot( dt, np.array(xpos)-xpos[fi],'ro', label='xpos' )
    ax[0].plot( dt, np.array(ypos)-ypos[fi],'bo', label='ypos' )
    ax[0].set_ylabel('Centroid Offset (px)')
    ax[0].legend(loc='best')

    ax[1].plot( dt, np.array(sigma),'ko' )
    ax[1].set_ylabel('PSF Sigma (px)')

    ax[2].plot( dt, bg,'ko')
    ax[2].set_ylabel('Average Background Flux / px')    
    
    ax[3].plot( dt, flux/np.mean(flux),'ko') 
    ax[3].set_ylabel('Relative Flux')

    ax[3].set_xlabel('Time (min)')

    plt.savefig("hatp16b_timeseries.png")
    plt.close()
