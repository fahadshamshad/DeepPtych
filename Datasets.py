import numpy as np
from glob import glob
import scipy.io as sio
from keras.datasets import mnist
from skimage.io import imread, imsave
# ---------------------- CONSTANTS -------------------------------

BLUR_PATH = './data/blur/blur_data.mat' # BLUR DATA PATH
SVHN_PATH = './data/svhn/*.mat' # SVHN DATA PATH
SVHN_TEST_SIZE = 30000


#----------------------- BLUR DATASET ----------------------------
class BlurDataSet():
    def __init__(self, path = BLUR_PATH):
        self.path = path
        self.Train = None
        self.Test = None
        
    def LoadData(self):
        Blur = sio.loadmat(self.path)
        Blur_Train_tmp = Blur['X_Train'][0]
        Blur_Test_tmp = Blur['X_Test'][0]
        Blur_Train = []
        for i in range(Blur_Train_tmp.shape[0]):
            Blur_Train.append(Blur_Train_tmp[i])
        Blur_Test = []
        for i in range(Blur_Test_tmp.shape[0]):
            Blur_Test.append(Blur_Test_tmp[i])    
        del Blur_Test_tmp, Blur_Train_tmp

        Blur_Train = np.array(Blur_Train)
        Blur_Test = np.array(Blur_Test)

        Blur_Train = np.expand_dims(Blur_Train, axis = -1)
        Blur_Test = np.expand_dims(Blur_Test, axis = -1)

        self.Train = Blur_Train
        self.Test  = Blur_Test

    def GetData(self):
        return self.Train, self.Test

    def GetBlur(self, idx = 'random', return_test = True):
        # TEST SET
        if return_test == True:
            if idx == 'all':
                return self.Test
            elif idx=='random':
                i = np.random.randint(self.Test.shape[0])
                return self.Test[i].reshape(28,28)
            else:
                return self.Test[idx].reshape(28,28)
        # TRAIN SET
        else:
            if idx == 'all':
                return self.Train
            elif idx=='random':
                i = np.random.randint(self.Train.shape[0])
                return self.Train[i].reshape(28,28)
            else:
                return self.Train[idx].reshape(28,28)

#----------------------- SVHN DATASET ----------------------------
SVHN_ORIG_IMAGES_RESULTS = './results/SVHN/Original Images/*.png'

class SVHNDataSet():
    def __init__(self, path = SVHN_PATH, TestSize = SVHN_TEST_SIZE):

        self.paths = glob(path)
        self.Train = None
        self.Test = None
        self.TestSize = TestSize
        self.IMAGES_IN_RESULT_FOLDER = None
    def LoadData(self):
        Data = np.zeros((531131,32,32,3))
        i = 0
        for path in self.paths:
            X = sio.loadmat(path)['X_BATCH']
            X = np.transpose(X, [3,0,1,2])
            Data[i:i+X.shape[0]] = X/255
            i = i + X.shape[0]
        self.Train = Data[:Data.shape[0]-self.TestSize]
        self.Test = Data[Data.shape[0]-self.TestSize:]


        
    def GetImage(self, idx = 'random', return_test = True):
        # TEST SET
        if return_test == True:
            if idx == 'all':
                return self.Test
            elif idx=='random':
                i = np.random.randint(self.Test.shape[0])
                return self.Test[i].reshape(32,32,3)
            else:
                return self.Test[idx].reshape(32,32,3)
        # TRAIN SET
        else:
            if idx == 'all':
                return self.Train
            elif idx=='random':
                i = np.random.randint(self.Train.shape[0])
                return self.Train[i].reshape(32,32,3)
            else:
                return self.Train[idx].reshape(32,32,3)

# -------------------------- MNSIT DATA ----------------------------

class MNISTDataSet():
    def __init__(self):
        self.Train = None
        self.Test = None
        
    def LoadData(self):
        (train, _ ), (test, _) = mnist.load_data() 
        self.Train = train.reshape(60000,28,28)
        self.Test = test.reshape(10000,28,28)

    def GetImage(self, idx = 'random', return_test = True):
        # TEST SET
        if return_test == True:
            if idx == 'all':
                return self.Test
            elif idx=='random':
                i = np.random.randint(self.Test.shape[0])
                return self.Test[i].reshape(28,28)
            else:
                return self.Test[idx].reshape(28,28)
        # TRAIN SET
        else:
            if idx == 'all':
                return self.Train
            elif idx=='random':
                i = np.random.randint(self.Train.shape[0])
                return self.Train[i].reshape(28,28)
            else:
                return self.Train[idx].reshape(28,28)

# ---------------------- EXPERIMENTS -------------------------------
Orig_Images_Path = glob('./results/svhn/Original Images/*.png')
Range_Images_Path = glob('/results/svhn/Range Images/*.png')

class EXPERIMENT_RESULTS():
    def __init__(self,path_recov_test, path_recov_range,
                 path_orig = Orig_Images_Path,
                 path_range  = Range_Images_Path):
        
        # LOADING ORIGINAL IMAGES
        path_orig = path_orig + '*.png'
        self.Orig  = np.array([imread(path) for path in path_orig])
        # LOADING RANGE IMAGES
        path_range = path_range + '*.png'
        self.Range  = np.array([imread(path) for path in path_range])
        # LOADING RANGE IMAGES
        path_recov_test = path_recov_test + '*.png'
        self.RecovFromTest  = np.array([imread(path) for path in path_recov_test])

        # LOADING RANGE IMAGES
        path_recov_range = path_recov_range + '*.png'
        self.RecovFromRange  = np.array([imread(path) for path in path_recov_range])

        # ANALYSIS VARIABLES
     
