from astropy.io import fits

path = "MPL-11/manga-7443-3703-1D.fits"

with fits.open(path) as hdul:
    header = hdul[0].header
    naxis = header.get("NAXIS", 0)
    
    print(f"NAXIS = {naxis}")
    
    if naxis >= 3:
        print("This is a data cube!")
    else:
        print("This is a 2D image or 1D spectrum")

with fits.open(path) as hdul:
    header = hdul[0].header
    naxis = header.get("NAXIS", 0)
    
    print(f"Number of dimensions: {naxis}")
    
    for i in range(1, naxis + 1):
        axis_size = header.get(f"NAXIS{i}")
        axis_type = header.get(f"CTYPE{i}", "unknown")
        print(f"  Axis {i}: size={axis_size}, type={axis_type}")

with fits.open(path) as hdul:
    data = hdul[0].data
    
    if data is None:
        print("No data in primary HDU")
    else:
        print(f"Data shape: {data.shape}")
        print(f"Number of dimensions: {data.ndim}")
        
        if data.ndim >= 3:
            print("This is a data cube!")
