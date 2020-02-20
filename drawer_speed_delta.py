import numpy as np
import matplotlib.pyplot as plt

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
#Batch Size = 512
plot_fid = True
plot_speed = False
dataset = 'cifar'
### CIFAR10:
# vq vae gated pixelcnn flops: 473147904

if dataset == 'cifar':
    adap_ar_ar_speed = np.array([[8.86, 19.10, 30.98, 44.34, 59.48],[47.61,96.52,148.71,182.07,222.73]])
    adap_ar_ar_speed[0] *= (220.63 / 59.48)
    adap_ar_total_speed = [3693.34, 2130.02, 1399.01, 1003.31, 776.68]  # it/s
    adap_ar_fid = np.array([118.49, 67.26, 56.34, 53.70, 53.62])
    adap_ar_fid = adap_ar_fid[1:]-adap_ar_fid[:-1]
    adap_ar_baseline_fid = np.array([[126.02, 79.69, 84.67, 80.87, 62.50]])
    adap_ar_baseline_fid = adap_ar_baseline_fid[:,1:] - adap_ar_baseline_fid[:,-1]
    adap_ar_vq_vae_fid = 53.86
    adap_ar_vq_vae_speed = [572.12] #499.64
    code_fraction = [0.4, 0.6, 0.8, 1.0]
elif dataset == 'celeba':
    #100000 samples
    adap_ar_ar_speed = np.array([[12.78, 28.70, 48.37, 71.89, 99.76], [13.33,30.06,49.62,73.01,101.88],[31.55,59.01,98.57,141.77,206.09]]) * (6.31)
    adap_ar_ar_speed[2] = adap_ar_ar_speed[2] / 2
    adap_ar_total_speed = []  # it/s
    adap_ar_fid = np.array([51.54,38.82,33.04,31.61,30.54])
    adap_ar_fid = adap_ar_fid[1:] - adap_ar_fid[:-1]
    adap_ar_baseline_fid = np.array([[150.62,79.40,41.61,36.86,34.88],[155.78,79.62,41.67,36.99,34.80]])

    adap_ar_baseline_fid = adap_ar_baseline_fid[:,1:] - adap_ar_baseline_fid[:,:-1]
    adap_ar_vq_vae_fid = 33.18
    adap_ar_vq_vae_speed = [1060.13] #978
    code_fraction = [0.4, 0.6, 0.8, 1.0]
if plot_fid:
    plt.plot(code_fraction, adap_ar_fid, label='Anytime AR', color='c')
    plt.plot(code_fraction, np.mean(adap_ar_baseline_fid,axis=0), label='Vanilla AR', color='y')
    plt.fill_between(code_fraction, adap_ar_fid + 1, adap_ar_fid - 1, facecolor='c', alpha=0.5)
    plt.fill_between(code_fraction,np.mean(adap_ar_baseline_fid,axis=0) + np.std(adap_ar_baseline_fid,axis=0), np.mean(adap_ar_baseline_fid,axis=0) - np.std(adap_ar_baseline_fid,axis=0), facecolor='y', alpha=0.5)
    #plt.hlines(y=adap_ar_vq_vae_fid, xmin=0.2, xmax=1.0, color='r', label='VQ-VAE', linestyles='dashed')
    plt.ylabel('$\Delta(i)$', font1)
    plt.xlabel('Fraction of dimensions', font1)
    if dataset == 'celeba':
        plt.title('CelebA', font1)
        plt.legend()
        #plt.show()
        plt.savefig('CelebA_fid_delta.jpg')
        plt.cla()
    else:
        plt.title('CIFAR-10', font1)
        plt.legend()
        #plt.show()
        plt.savefig('CIFAR-10_fid_delta.jpg')
        plt.cla()
if plot_speed:
    mu = np.mean(adap_ar_ar_speed, axis=0)
    var = np.std(adap_ar_ar_speed, axis=0)
    plt.plot(code_fraction, mu, label='Adaptive AR', color='c')
    #plt.plot(code_fraction, adap_ar_vq_vae_speed, label='Baseline', color='y')
    plt.fill_between(code_fraction, mu + var, mu - var, facecolor='c', alpha=0.5)
    plt.hlines(y=adap_ar_vq_vae_speed[0], xmin=0.2, xmax=1.0, color='r', label='VQ-VAE', linestyles='dashed')
    plt.ylabel('Time (seconds)', font1)
    plt.xlabel('Fraction of dimensions', font1)
    if dataset == 'celeba':
        plt.title('CelebA', font1)
        plt.legend()
        #plt.show()
        plt.savefig('CelebA_speed.jpg')
    else:
        plt.title('CIFAR-10', font1)
        plt.legend()
        #plt.show()
        plt.savefig('CIFAR-10_speed.jpg')
