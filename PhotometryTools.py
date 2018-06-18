import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.optimize import least_squares,minimize
from scipy.interpolate import RectBivariateSpline

def star_psf(x,y,x0,y0,a,sigx,sigy,b):
    gaus = a * np.exp(-(x-x0)**2 / (2*sigx**2) ) * np.exp(-(y-y0)**2 / (2*sigy**2) ) + b
    return gaus

class psf(object):
    def __init__(self,x0,y0,a,sigx,sigy,b,rot=0):
        self.pars = [x0,y0,a,sigx,sigy,b]
        self.a = a
        self.x0 = x0
        self.y0 = y0
        self.sigx = sigx
        self.sigy = sigy
        self.b = b
        self.rot = rot

    def eval(self,x,y):
        if self.rot == 0:
            return star_psf(x,y,*self.pars)
        else:
            return rotate(star_psf(x,y,*self.pars),self.rot,reshape=False)

    @property
    def gaussian_area(self):
        # PSF area without background
        return 2*np.pi*self.a*self.sigx*self.sigy

    @property
    def cylinder_area(self):
        # models background 
        return np.pi*(3*self.sigx * 3*self.sigy) * self.b
        
    @property
    def area(self):
        return self.gaussian_area + self.cylinder_area


class ccd(object):
    def __init__(self,size):

        if isinstance(size,np.ndarray): # load data from array
            self.data = np.copy(size)
        else:
            self.data = np.zeros(size)

    def draw(self,star):
        b=max(star.sigx,star.sigy)*5
        x = np.arange( int(star.x0-b), int(star.x0+b+1) )
        y = np.arange( int(star.y0-b), int(star.y0+b+1) )
        xv, yv = np.meshgrid(x, y)
        self.data[yv,xv] += star.eval(xv,yv)

def mesh_box(pos,box,mesh=True,npts=-1):
    pos = [int(np.round(pos[0])),int(np.round(pos[1]))]
    if npts == -1:
        x = np.arange(pos[0]-box, pos[0]+box+1)
        y = np.arange(pos[1]-box, pos[1]+box+1)
    else:
        x = np.linspace(pos[0]-box, pos[0]+box+1,npts)
        y = np.linspace(pos[1]-box, pos[1]+box+1,npts)

    if mesh:
        xv, yv = np.meshgrid(x, y)
        return xv,yv
    else:
        return x,y

def estimate_sigma(x,maxidx=-1):
    if maxidx == -1:
        maxidx = np.argmax(x)
    lower = np.abs(x-0.5*np.max(x))[:maxidx].argmin()
    upper = np.abs(x-0.5*np.max(x))[maxidx:].argmin()+maxidx
    FWHM = upper-lower
    return FWHM/(2*np.sqrt(2*np.log(2)))

def fit_centroid(data,pos,init=None,psf_output=False,lossfn='linear',box=25):
    if not init: # if init is none, then set the values
        init=[-1,5,5,0]

    # estimate the amplitude and centroid
    if init[0]==-1:
        # subarray of data around star
        xv,yv = mesh_box(pos,box)

        # amplitude guess
        init[0] = np.max( data[yv,xv] )

        # weighted sum to estimate center
        wx = np.sum(np.unique(xv)*data[yv,xv].sum(0))/np.sum(data[yv,xv].sum(0))
        wy = np.sum(np.unique(yv)*data[yv,xv].sum(1))/np.sum(data[yv,xv].sum(1))
        pos = [wx, wy]
        # estimate std by calculation of FWHM
        x,y= data[yv,xv].sum(0),data[yv,xv].sum(1)
        init[1] = estimate_sigma(x)
        init[2] = estimate_sigma(y)

        # bg estimate
        # compute the average from 1/4 of the lowest values in the bg
        init[3] = np.mean( np.sort( data[yv,xv].flatten() )[:int(data[yv,xv].flatten().shape[0]*0.25)] )
    #print('init pos:',pos)
    #print('init2:',init)

    # recenter data on weighted average of light
    xv,yv = mesh_box( pos ,box)

    # pars = x,y, a,sigx,sigy, rotate
    def fcn2min(pars):
        model = star_psf(xv,yv,*pars)
        return (data[yv,xv]-model).flatten() # method for LS
        #return np.sum( (data[yv,xv]-model)**2 ) # method for minimize

    # TODO make these inputs to function?
    lo = [pos[0]-box,pos[1]-box,0,1,1,0]
    up = [pos[0]+box,pos[1]+box,64000,40,40,np.max(data[yv,xv])]
    res = least_squares(fcn2min,x0=[*pos,*init],bounds=[lo,up],loss=lossfn,jac='3-point')
    #res = minimize(fcn2min,x0=[*pos,*init],method='Nelder-Mead')
    del init

    if psf_output:
        return psf(*res.x,0)
    else:
        return res.x

def circle_mask(x0,y0,r=25,samp=10):
    xv,yv = mesh_box([x0,y0],r+1,npts=samp)
    rv = ((xv-x0)**2 + (yv-y0)**2)**0.5
    mask = rv<r
    return xv,yv,mask

def sky_annulus(x0,y0,r=25,dr=5,samp=10):
    xv,yv = mesh_box([x0,y0],r+dr+1,npts=samp)
    rv = ((xv-x0)**2 + (yv-y0)**2)**0.5
    mask = (rv>r) & (rv<(r+dr)) # sky annulus mask
    return xv,yv,mask

def phot(x0,y0,data,r=25,dr=5,samp=3,debug=False,bgsub=False):

    # determine img indexes for aperture region
    xv,yv = mesh_box([x0,y0],r+2)

    # derive indexs on a higher resolution grid and create aperture mask
    px,py,mask = circle_mask(x0,y0,r=r,samp=xv.shape[0]*samp)

    # interpolate original data onto higher resolution grid
    subdata = data[yv,xv]
    model = RectBivariateSpline(np.unique(xv),np.unique(yv),subdata)

    # evaluate data on highres grid
    pz = model.ev(px,py)

    # zero out pixels larger than radius
    pz[~mask] = 0

    # subtract off the background
    if isinstance(bgsub,bool):
        # get the bg flux per pixel
        bgflux = skybg_phot(x0,y0,data,r,dr,samp)
    else:
        #assume the bgsub value from the centroid fit
        bgflux = bgsub

    # sum over circular aperture and subtract bg flux from each pixel
    pz -= bgflux

    # remove negative pixel values
    pz[pz<0] = 0

    # scale area back to original grid
    parea = pz.sum()*np.diff(px).mean()*np.diff(py[:,0]).mean()


    if debug:
        print('   mask area=',mask.sum()*np.diff(px).mean()*np.diff(py[:,0]).mean()  )
        print('cirular area=',np.pi*r**2)
        print('square aper =',subdata.sum()) # square aperture sum
        print('   phot flux=',parea)
        print('bg flux/pix =',bgflux)
        totalbg = bgflux*np.diff(px).mean()*np.diff(py[:,0]).mean()*mask.sum()
        print('     bg flux=',totalbg )
        import pdb; pdb.set_trace()

    return parea

def skybg_phot(x0,y0,data,r=25,dr=5,samp=3,debug=False):

    # determine img indexes for aperture region
    xv,yv = mesh_box([x0,y0],(r+dr)+2)

    # derive indexs on a higher resolution grid and create aperture mask
    px,py,mask = sky_annulus(x0,y0,r=r,samp=xv.shape[0]*samp)

    # interpolate original data onto higher resolution grid
    subdata = data[yv,xv]
    model = RectBivariateSpline(np.unique(xv),np.unique(yv),subdata)

    # evaluate data on highres grid
    pz = model.ev(px,py)

    # zero out pixels larger than radius
    pz[~mask] = 0
    pz[pz<0] = 0

    # scale area back to original grid, total flux in sky annulus
    parea = pz.sum() * np.diff(px).mean()*np.diff(py[:,0]).mean()

    if debug:
        print('mask area=',mask.sum()*np.diff(px).mean()*np.diff(py[:,0]).mean()  )
        print('true area=',2*np.pi*r*dr)
        print('subdata flux=',subdata.sum())
        print('bg phot flux=',parea)
        import pdb; pdb.set_trace()

    # return bg value per pixel
    return pz.sum()/mask.sum()


if __name__ == "__main__":

    img = ccd([1024,1024])
    star = psf(256,512,2000,4,4,0,0)
    img.draw(star)

    pars_psf = fit_centroid(img.data,[250,506],box=25,psf_output=False)
    area = phot(pars_psf[0],pars_psf[1],img.data,r=15,debug=False,bgsub=True)
    print('best fit parameters:',pars_psf)
    print('phot area=',area)
    print('psf area=',star.gaussian_area)