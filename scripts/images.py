# Tested Python 3.6

import webbrowser
import urllib.request
from astropy.io import fits

file='dataset.fits'

hdulist=fits.open(file)
tbdata = hdulist[1].data
name = tbdata.field('name')
ra = tbdata.field('ra')
dec = tbdata.field('dec')

for i in range(len(ra)):

    scale = 0.262

# Download from DESI
   
    url1 = 'http://legacysurvey.org/viewer/jpeg-cutout?ra=%f&dec=%f&size=800&layer=ls-dr10&pixscale=%f&bands=grz' % (ra[i], dec[i],scale)

# Save images from url
    urllib.request.urlretrieve(url1, name[i]+'.jpg')

    print(i)
