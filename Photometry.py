import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.optimize import least_squares

from photutils import aperture_photometry, CircularAperture, CircularAnnulus

def mixed_psf(x,y,x0,y0,a,sigx,sigy,rot,b, w):
    gaus = gaussian_psf(x,y,x0,y0,a,sigx,sigy,rot, 0)
    lore = lorentz_psf(x,y,x0,y0,a,sigx,sigy,rot, 0)
    return (1-w)*gaus + w*lore + b

def gaussian_psf(x,y,x0,y0,a,sigx,sigy,rot, b):
    rx = (x-x0)*np.cos(rot) - (y-y0)*np.sin(rot)
    ry = (x-x0)*np.sin(rot) + (y-y0)*np.cos(rot)
    gausx = np.exp(-(rx)**2 / (2*sigx**2) )
    gausy = np.exp(-(ry)**2 / (2*sigy**2) )
    return a*gausx*gausy + b

def lorentz_psf(x,y,x0,y0,a,sigx,sigy,rot, b):
    rx = (x-x0)*np.cos(rot) - (y-y0)*np.sin(rot)
    ry = (x-x0)*np.sin(rot) + (y-y0)*np.cos(rot)
    lorex = sigx**2 / ( rx**2 + sigx**2 )
    lorey = sigy**2 / ( ry**2 + sigy**2 )
    return a*lorex*lorey + b

class psf(object):
    def __init__(self,pars,psf_function):
        self.pars = pars # x0,y0,a,sigx,sigy,rot
        self.fn = psf_function

    def eval(self,x,y):
        return self.fn(x,y,*self.pars)
        
class ccd(object):
    def __init__(self,size):

        if isinstance(size,np.ndarray): # load data from array
            self.data = np.copy(size)
        else:
            self.data = np.zeros(size)

    def draw(self,star):
        b=max(star.pars[3],star.pars[4])*5
        x = np.arange( int(star.pars[0]-b-1), int(star.pars[0]+b+1) ) 
        y = np.arange( int(star.pars[1]-b-1), int(star.pars[1]+b+1) )
        # TODO constrain evaluation bounds to within image?
        xv, yv = np.meshgrid(x, y)
        self.data[yv,xv] += star.eval(xv,yv)

def mesh_box(pos,box):
    pos = [int(np.round(pos[0])),int(np.round(pos[1]))]
    x = np.arange(pos[0]-box, pos[0]+box+1)
    y = np.arange(pos[1]-box, pos[1]+box+1)
    xv, yv = np.meshgrid(x, y)
    return xv.astype(int),yv.astype(int)
    
def fit_psf(data,pos,init,lo,up,psf_function=gaussian_psf,lossfn='linear',box=15):
    xv,yv = mesh_box(pos, box)
    def fcn2min(pars):
        model = psf_function(xv,yv,*pars)
        return (data[yv,xv]-model).flatten()
    res = least_squares(fcn2min,x0=[*pos,*init],bounds=[lo,up],loss=lossfn,jac='3-point')
    return res.x

def phot(data,xc,yc,r=25,dr=5):

    if dr>0:
        bgflux = skybg_phot(data,xc,yc,r,dr)
    else:
        bgflux = 0
    positions = [(xc, yc)]
    data = data-bgflux
    data[data<0] = 0 

    try:
        apertures = CircularAperture(positions, r=r)
        phot_table = aperture_photometry(data, apertures, method='exact')
        # apertures.to_mask()[0].data
        return float(phot_table['aperture_sum']) 
    except:
        # create high res mask (TODO make more efficient, precompute mask + pass in)
        xvh,yvh = mesh_box([xc,yc], (np.round(r)+1)*10 )
        rvh = ((xvh-xc)**2 + (yvh-yc)**2)**0.5
        maskh = (rvh<r*10)

        # downsize to native resolution 
        xv,yv = mesh_box([xc,yc], (np.round(r)+1) )
        mask = imresize(maskh, xv.shape) # rough approx for fractional pixels
        mask = mask / mask.max()
        return np.sum(data[yv,xv] * mask)

def skybg_phot(data,xc,yc,r=25,dr=5):    
    # create a crude annulus to mask out bright background pixels 
    xv,yv = mesh_box([xc,yc], np.round(r+dr) )
    rv = ((xv-xc)**2 + (yv-yc)**2)**0.5
    mask = (rv>r) & (rv<(r+dr))
    cutoff = np.percentile(data[yv,xv][mask], 50)
    dat = np.copy(data)
    dat[dat>cutoff] = cutoff # ignore bright pixels like stars 

    try:
        # under estimate background 
        positions = [(xc, yc)]
        apertures = CircularAnnulus(positions, r_in=r, r_out=r+dr)
        phot_table = aperture_photometry(dat, apertures, method='exact')
        aper_area = apertures.area()
        return float(phot_table['aperture_sum'])/aper_area
    except:
        return min( np.mean(data[yv,xv][mask]), np.median(data[yv,xv][mask]) )

def estimate_sigma(x,maxidx=-1):
    if maxidx == -1:
        maxidx = np.argmax(x)
    lower = np.abs(x-0.5*np.max(x))[:maxidx].argmin()
    upper = np.abs(x-0.5*np.max(x))[maxidx:].argmin()+maxidx
    FWHM = upper-lower
    return FWHM/(2*np.sqrt(2*np.log(2)))



if __name__ == "__main__":

    # simulate an image
    img = ccd([32,32])
    star = psf( [15.1,15.0, 1000, 1.5, 1, np.pi/6, 0], gaussian_psf)
    img.data += 10*np.random.random( img.data.shape)
    img.draw(star)
    plt.imshow(img.data)
    plt.show()

    # compute flux weighted centroid on subarray
    xv,yv = mesh_box([15,15],5)
    wx = np.sum(np.unique(xv)*img.data[yv,xv].sum(0))/np.sum(img.data[yv,xv].sum(0))
    wy = np.sum(np.unique(yv)*img.data[yv,xv].sum(1))/np.sum(img.data[yv,xv].sum(1))

    # estimate standard deviation 
    x,y= img.data[yv,xv].sum(0),img.data[yv,xv].sum(1) 
    sx = estimate_sigma(x)
    sy = estimate_sigma(y)

    # fit PSF 
    pars = fit_psf(
        img.data,
        [wx, wy], 
        [np.max(img.data[yv,xv]), sx, sy, 0, np.min(img.data[yv,xv]) ], # initial guess: [amp, sigx, sigy, rotation, bg]
        [wx-5, wy-5, 0,   0, 0, -np.pi/4, 0],                           # lower bound: [xc, yc, amp, sigx, sigy, rotation,  bg]
        [wx+5, wy+5, 1e5, 2, 2,  np.pi/4, np.percentile(img.data,25)],  # upper bound: 
        psf_function=gaussian_psf,
        box=5 # only fit a subregion +/- 5 px from centroid
    )

    area = phot(img.data, pars[0],pars[1],r=2.5,dr=8)
    print('best fit parameters:',pars)
    print('phot area=',area)
    print('psf area=',2*np.pi*pars[2]*pars[3]*pars[4])
    print('true area=',2*np.pi*star.pars[2]*star.pars[3]*star.pars[4])

    # compute PSF fit residual
    xv,yv = mesh_box([15,15], 5) # pull out subregion that was fit 
    model = psf( pars, gaussian_psf).eval(xv,yv)
    residual = img.data[yv,xv] - model 

    # diagnostic plots
    f,ax = plt.subplots(1,3)
    ax[0].imshow(img.data[yv,xv]); ax[0].set_title('Raw Data')
    ax[1].imshow(model); ax[1].set_title('PSF model')
    ax[2].imshow(residual); ax[2].set_title('Residual')
    plt.show()

    dude() 
    # simulate some wierd pixel sensitivity 
    # xv,yv = mesh_box([15,15], 15 )
    # pr = ( (xv-15) + 5*(yv-15) )**2 + ( 5*(xv-15) + (yv-15) +3 )**2 * 0.5*np.cos(xv) * np.sin(yv)
    # pr = pr/pr.max()
    # pr *= -1
    # pr += 1
    # plt.imshow(pr)
    # plt.show()

    NPTS = 1000

    # simulate transit data 
    from ELCA import lc_fitter, transit
    from flux_decorrelation import gaussian_weights 

    t = np.linspace(0.85,1.05,NPTS)
    init = { 'rp':0.1, 'ar':14.07,       # Rp/Rs, a/Rs
             'per':3.336817, 'inc':88.75, # Period (days), Inclination
             'u1': 0.3, 'u2': 0,          # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,            # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,              # Airmass extinction terms
             'a2':0, 'tm':0.95 }          # tm = Mid Transit time (Days)
    data = transit(time=t, values=init) #+ np.random.normal(0, 2e-4, len(t))

    # simulate PSFs on detector
    xcent = [15.13]
    ycent = [14.96]
    sigmax = [0.55]
    sigmay = [0.55]

    for i in range(1,NPTS):
        xcent.append( xcent[i-1] + np.random.normal(0,0.005) )
        ycent.append( ycent[i-1] + np.random.normal(0,0.005) )
        sigmax.append( sigmax[i-1] + np.random.normal(0,0.0005) )
        sigmay.append( sigmay[i-1] + np.random.normal(0,0.0005) )

    # simulate images on detector 
    X = np.zeros((NPTS,7))
    fluxs = np.zeros((NPTS,4))

    images = []
    photflux = []
    sqrflux = []
    trueflux = [] 
    psfflux = []
    wxc = []
    wyc = []

    for i in range(NPTS):

        img = ccd([32,32])
        fluxscale = (sigmax[0]*sigmay[0]) / (sigmax[i]*sigmay[i])

        star = psf( 
            [ 
                xcent[i], ycent[i],
                1000*fluxscale*data[i],
                sigmax[i], sigmay[i],
                np.pi/6, 0 # TODO fix rotation changing area of Gaussian
            ], 
            gaussian_psf
        )
        img.draw(star)
        img.data += np.random.random( img.data.shape)
        images.append( img.data)

        # compute flux weighted centroid on subarray
        xv,yv = mesh_box([np.argmax(img.data.sum(0)),np.argmax(img.data.sum(1))],5)
        wx = np.sum(np.unique(xv)*img.data[yv,xv].sum(0))/np.sum(img.data[yv,xv].sum(0))
        wy = np.sum(np.unique(yv)*img.data[yv,xv].sum(1))/np.sum(img.data[yv,xv].sum(1))
        wxc.append( wx); wyc.append( wy)

        # estimate standard deviation 
        x,y= img.data[yv,xv].sum(0),img.data[yv,xv].sum(1) 
        sx = estimate_sigma(x)
        sy = estimate_sigma(y)

        # fit PSF 
        pars = fit_psf(
            img.data,
            [wx, wy], 
            [np.max(img.data[yv,xv]), sx, sy, 0, np.min(img.data[yv,xv]) ], # initial guess: [amp, sigx, sigy, rotation, bg]
            [wx-5, wy-5, 0,   0, 0, -np.pi/4, 0],                           # lower bound: [xc, yc, amp, sigx, sigy, rotation,  bg]
            [wx+5, wy+5, 1e5, 2, 2,  np.pi/4, np.percentile(img.data,25)],  # upper bound: 
            psf_function=gaussian_psf,
            box=5 # only fit a subregion +/- 5 px from centroid
        )

        X[i] = pars
        psfflux.append( 2*np.pi*pars[2]*pars[3]*pars[4] )
        trueflux.append( fluxscale*2*np.pi*1000*sigmax[i]*sigmay[i]*data[i] )
        aperphot = phot(img.data, wx, wy, r=2.5,dr=8)
        photflux.append( aperphot)        

    # test decorrelation methods 
    gw, nearest = gaussian_weights( X[:,:2] )


    flux = np.array(photflux)/np.mean(photflux)
    wf = np.array([np.sum(flux[nearest[i]] * gw[i]) for i in range(len(flux))])


    f,ax = plt.subplots(3)
    ax[0].plot(t,xcent,label='X true',alpha=0.5)
    ax[0].plot(t,ycent,label='Y true',alpha=0.5)
    ax[0].plot(t,X[:,0],marker='.',ls='none',label='X est',alpha=0.5)
    ax[0].plot(t,X[:,1],marker='.',ls='none',label='Y est',alpha=0.5)
    ax[0].plot(t,wxc,marker='.',ls='none',label='X fw',alpha=0.5)
    ax[0].plot(t,wyc,marker='.',ls='none',label='Y fw',alpha=0.5)

    ax[0].set_ylabel('Centroid Position [px]')
    ax[0].legend(loc='best')
    ax[1].plot(t,sigmax,label='X true',alpha=0.5)
    ax[1].plot(t,sigmay,label='Y true',alpha=0.5)
    ax[1].plot(t,X[:,3],marker='.',ls='none',label='X est',alpha=0.5)
    ax[1].plot(t,X[:,4],marker='.',ls='none',label='Y est',alpha=0.5)
    ax[1].legend(loc='best')
    ax[1].set_ylabel('PSF sigma [px]')
    ax[2].plot(t,np.array(photflux)/np.mean(photflux),'g.',label='Aperture Phot',alpha=0.5)
    #ax[2].plot(t,np.array(photflux)/np.mean(photflux)/wf,'r.',label='Aperture Phot + GW Decorrelation',alpha=0.5)
    ax[2].plot(t,np.array(psfflux )/np.mean(psfflux),'c.',label='PSF Phot',alpha=0.5)
    ax[2].plot(t,np.array(trueflux)/np.mean(trueflux),'k.',label='Truth',alpha=0.5)
    ax[2].set_xlabel('Time')
    ax[2].legend(loc='best')
    plt.show()