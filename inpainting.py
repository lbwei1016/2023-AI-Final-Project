from generate_image import generate_images

network = "./pretrained/CelebA-HQ_512.pkl"
dpath = "./test_sets/images"
mpath = "./test_sets/masks"
outdir = "./test_sets/images"
resolution = 1024

def inpaint():
    generate_images(
        network_pkl=network, 
        dpath=dpath, 
        mpath=mpath, 
        outdir=outdir, 
        noise_mode='random', 
        resolution=resolution
    )

