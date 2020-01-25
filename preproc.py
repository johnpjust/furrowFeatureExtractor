from skimage import io, transform, filters, color
import glob
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter

class preproc_imgs:

    def __init__(self,
                 path=None):
        #### read in data directory of images
        self.IMsizereduce = 0.1

        # listing = glob.glob(r'Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\West Bilsland Left Wing Log 17\*.png')
        ## listing = dir('Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\West Bilsland Left Wing Log 15\*.png');
        ## listing = dir('Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\West Bilsland Left Wing Log 19\*.png');
        ## listing = dir('Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\Bennett Left Wing Log 9\*.png');
        ## listing = dir('Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\East Bilsland Left Wing Log 3\*.png');
        ## listing = dir('Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\Uthe Left Wing Log 2\*.png');

        #### get the first image to set things up
        self.quantileVal = 0.90
        I = io.imread(path)
        Irs = transform.rescale(I, self.IMsizereduce)
        Irs = Irs[:,30:130,:] ## cut off edges where black -- loses come context due to angled mask
        light_scalar = 80/np.double(np.quantile(Irs.reshape((-1)), self.quantileVal))
        ###### background = imread('C:\Users\justjo\Desktop\background.png');
        Irs_res = np.uint8(np.double(Irs)*light_scalar)
        ###### subtract off background vignetting
        background = filters.gaussian(np.double(Irs_res),30)
        I2 = np.double(Irs_res)-np.double(background)
        I2 = np.uint8(I2 - np.min(I2.reshape((-1))))
        #### setup trench image mask for location
        edge = 20/2
        # x1bot = 123 - edge
        # x1top = 126 - edge
        # x2bot = 195 + edge
        # x2top = 199 + edge
        x1bot = (123-60)//2 - edge
        x1top = (126-60)//2 - edge
        x2bot = (195-60)//2 + edge
        x2top = (199-60)//2 + edge

        self.imsize = I2.shape[:2]
        ysize, xsize, _ = I2.shape
        self.ymat = np.arange(0,ysize)
        self.xmat = np.arange(0,xsize)

        self.leftfur = self.ymat*(x1bot-x1top)/(I2.shape[0]) + x1top
        self.rightfur = self.ymat*(x2bot-x2top)/(I2.shape[0]) + x2top

        self.mask_fur = np.zeros((ysize, xsize))
        for yn in np.arange(0,ysize):
            self.mask_fur[yn, :] = np.logical_and(self.xmat < self.rightfur[yn],  self.xmat > self.leftfur[yn])

        self.thediff = np.int(self.rightfur[0] - self.leftfur[0])

    def proc_imgs(self, path):

        I = io.imread(path)
        Irs = transform.rescale(I, self.IMsizereduce)
        Irs = Irs[:,30:130,:] ## cut off edges where black -- loses come context due to angled mask
        light_scalar = 80/np.double(np.quantile(Irs.reshape((-1)), self.quantileVal))
        ###### background = imread('C:\Users\justjo\Desktop\background.png');
        Irs_res = np.uint8(np.double(Irs)*light_scalar)
        ###### subtract off background vignetting
        background = filters.gaussian(np.double(Irs_res), 30)
        I2 = np.double(Irs_res) - np.double(background)
        I2 = np.uint8(I2 - np.min(I2.reshape((-1))))

        knum = 4
        R = I2[:,:, 0].reshape(-1)
        G = I2[:,:, 1].reshape(-1)
        B = I2[:,:, 2].reshape(-1)
        rgb = np.array([R, G, B]).T
        C = KMeans(knum, max_iter=500).fit(np.double(rgb))

        # self.C = C ## make sure this is converging

        VAL = np.mean(C.cluster_centers_,axis=1)
        IND = np.argmin(VAL)

        ## ## find ground labels
        trench = C.labels_.reshape(I2.shape[:2])
        trench[trench != IND] = -1
        trench[trench == IND] = 1
        trench[trench == -1] = 0

        ## ## ## ## sgolay trench edge finder ## ## ## ##
        trench_feat = np.zeros((self.thediff + 1, 1))
        for n in range(self.thediff + 1):
            linearr = np.round((self.leftfur * (self.thediff - n + 1) + self.rightfur * (n - 1)) / self.thediff, 0).astype(np.int)
            trench_feat[n] = np.mean(trench[self.ymat, linearr])

        trench_feat_filt_diff = np.diff(savgol_filter(trench_feat.squeeze(), 13, 2))
        zci = lambda v: np.where(v * np.roll(v, [-1, 0]) <= 0)[0]
        trench_feat_zero_locs = zci(trench_feat_filt_diff)

        trenchmag_thresh = 0.15
        mag_change = trench_feat[trench_feat_zero_locs[1:]] - trench_feat[trench_feat_zero_locs[0:-1]]
        trench_left_loc = 10
        trench_right_loc = len(trench_feat) - 10
        temp = np.where(mag_change > trenchmag_thresh)[0]
        if np.size(temp) > 0:
            trench_left_loc = trench_feat_zero_locs[temp[0]]

        temp = np.where(-mag_change > trenchmag_thresh)[0]
        if np.size(temp) > 0:
            trench_right_loc = trench_feat_zero_locs[temp[-1] + 1]

        ## ## ## if edges are messed up then swap them
        if trench_left_loc >= trench_right_loc:
            temp = trench_left_loc
            trench_left_loc = trench_right_loc
            trench_right_loc = temp

        ## (1) get trench edge locations
        ## (2) calculate trench width, avg value in trench, avg value outside of trench
        ## (3)store in vectors as signal for log
        trench_inside_tr_meanperc_signal = np.median(trench_feat[trench_left_loc:trench_right_loc])
        ## ## ## median
        trench_outside_tr_meanperc_signal = np.median(np.vstack((trench_feat[1:trench_left_loc], trench_feat[trench_right_loc:])))

        median_fur_quality = trench_inside_tr_meanperc_signal - trench_outside_tr_meanperc_signal

        return median_fur_quality, I2