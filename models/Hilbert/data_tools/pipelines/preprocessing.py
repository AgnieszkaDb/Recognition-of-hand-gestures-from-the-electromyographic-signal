import numpy as np
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin

class TrfLowpass(BaseEstimator, TransformerMixin):
    def __init__(self, f, fs, order):
        self.f = f
        self.fs = fs
        self.order = order
        b, a = scipy.signal.butter(N=self.order, Wn=self.f, fs=self.fs, btype='lowpass', output='ba')
        self.b = b
        self.a = a
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        output = np.copy(X)
       # X = np.array(X)        
        for i in range(len(X)):
            output[i] = self.lpf(X[i])
        return output
    
    def lpf(self, x):
        return scipy.signal.filtfilt(self.b, self.a, x)

class TrfNormalize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._min = 0
        self._max = 0

    def fit(self, X, y=None):
        # Ensure X is a NumPy array
        X = np.array(X)
        print("shape is:", (X.shape))        

        print("nr of dim: ", X.ndim, "type: ", type(X))
                
        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)
        return self

    def transform(self, X, y=None):
        print(type(X))
        output = np.copy(X)
        for i in range(X.shape[0]):
            output[i] = self.normalize(X[i])
        return output
    
    def normalize(self, x):
        return (x - self._min) / (self._max - self._min)
