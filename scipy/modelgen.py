import numpy as np

def parameterMatrix(length, changePointPos, w1, w2, sampleFreq = 8000):
    
    """ Generate the parameter matrix G in the General Linear Model: 
  
    d = Gb + e

    Parameters
    ----------

    length = kernel length, int
    changePointPos = index of the frequency jump within the kernel, int
    w1 = lower frequency in Hz, float64
    w2 = higher frequency in Hz, float64
    sampleFreq = sampling frequency in Hz, optional (=8000), float64


    """

    # Initialise the matrix with the correct dimension
    G = np.zeros((length, 2))

    # Fill up G with the chunk of sinusoid at lower freq w1
    for i in range(0, changePointPos):
        G[i,0] = np.sin( (w1 * (2*np.pi/sampleFreq)) * i)
        G[i,1] = np.cos( (w1 * (2*np.pi/sampleFreq)) * i)
    
    # Fill up G with the remaining chunk of sinusoid at higher freq w2
    for j in range(changePointPos+1, length):
        G[j,0] = np.sin( (w2 * (2*np.pi/sampleFreq)) * j)
        G[j,1] = np.cos( (w2 * (2*np.pi/sampleFreq)) * j)

    return G


def fastParameterMatrix(length, changePointPos, w1, w2, sampleFreq = 8000):
    
    """Same as paramaterMatrix() but runs > 10 times faster
    note: code less obvious. see:
    http://docs.scipy.org/doc/numpy/reference/routines.indexing.html
    
    Generate the parameter matrix G in the General Linear Model: 

    d = Gb + e

    Parameters
    ----------

    length = kernel length, int
    changePointPos = index of the frequency jump within the kernel, int
    w1 = lower frequency in Hz, float64
    w2 = higher frequency in Hz, float64
    sampleFreq = sampling frequency in Hz, optional (=8000), float64


    """

    index = np.arange(0, length)

    freqnorm1 = w1 * (2 * np.pi/sampleFreq)
    freqnorm2 = w2 * (2 * np.pi/sampleFreq)

    G = np.r_[ np.c_[np.sin(freqnorm1*index[:changePointPos]),
                     np.cos(freqnorm1*index[:changePointPos])], 
               np.c_[np.sin(freqnorm2*index[changePointPos+1:]), 
                     np.cos(freqnorm2*index[changePointPos+1:])] ]

    return G
