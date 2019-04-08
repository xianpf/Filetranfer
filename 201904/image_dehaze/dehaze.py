import cv2;
import math;
import numpy as np;

def DarkChannel(im,sz):
    # b,g,r = cv2.split(im)
    # dc = cv2.min(cv2.min(r,g),b);
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    # dark = cv2.erode(dc,kernel)
    # return dark
    dark = np.zeros(im.shape[:-1])
    pad_w = math.floor(sz/2)
    padded_image = np.pad(im, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)), 'edge')

    for i, j in np.ndindex(im.shape[:-1]):
        dark[i, j] = np.min(padded_image[i:i + sz, j:j + sz, :])  
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    # numpx = int(max(math.floor(imsz/1000),1))
    numpx = int(0.01 * imsz)
    darkvec = dark.reshape(1,imsz);

    

    indices = darkvec.argsort();
    max_10_per = indices[0,imsz-numpx:]
    A = np.zeros((1,3))
    for j in range(3):
        for i in range(numpx):
            x = int(max_10_per[i] / w)
            y = max_10_per[i] % w
            A[0,j] += im[x,y,j]
    A = A / numpx

    # imvec = im.reshape(imsz,3);
    # atmsum = np.zeros([1,3])
    # for ind in range(1,numpx):
    #    atmsum = atmsum + imvec[indices[ind]]

    # A = atmsum / numpx;
    # A = np.array([[1, 1, 1]])
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 600;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '1.jpg'

    def nothing(*argv):
        pass

    src = cv2.imread('./image/'+fn);

    I = src.astype('float64')/255;
 
    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.3);
    J = J * 1.1 + 0.05

    # cv2.imshow("dark",dark);
    # cv2.imshow("t",t);
    # cv2.imshow('I',src);
    # cv2.imshow('J',J);
    cv2.imwrite('./image/'+'1_p.jpg',J*255);
    cv2.waitKey();
    
