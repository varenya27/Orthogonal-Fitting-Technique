import emcee
import numpy as np
from matplotlib import pyplot as plt
import time 
import pandas as pd
import pickle 
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import corner
import os

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

#priors for x, y and intrinsic scatter:
min_ = -6.
max_ = 12.
min_scat = -0.
max_scat = 1.

def logprior(theta):
    m, c, scat_int = theta
    if min_<m<max_ and min_<c<max_ and min_scat<scat_int<max_scat:
        return 0
    else: 
        return -np.inf

#vertical and orthogonal likelihoods defined below
#by default, the code executes the orthogonal likelihood
def loglikelihood1(theta, y, x, err_y, err_x):
    # likelihood L1 vertical:
    m, c, sigma_int = theta
    sigma2 = err_y**2+(m*err_x)**2 + sigma_int**2
    md = straight_line(theta,x)
    return  -0.5 * np.sum( (y-md)**2/sigma2 + np.log(2*np.pi*sigma2))

def loglikelihood2(theta, y, x, err_y, err_x):
    # likelihood L2 orthogonal:
    m, c, sigma_int = theta
    sigma2 = ((m**2)*err_x**2)/(m**2+1)+(err_y**2)/(m**2+1)+sigma_int**2
    md = straight_line(theta,x)
    delta = ((y-md)**2) / (m**2+1)
    return -0.5 * np.sum(np.log(2*np.pi*sigma2)+(delta/(sigma2)))

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) 
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood2(theta, y, x, err_y, err_x)

#reading data from a test file
df = pd.read_csv('data.csv') 
y, err_y, x, err_x= df.logMstar, df.logMstar_err, df.logV, df.logV_err

#initializing paramters for the sampler
Nens = 300 # number of ensemble points
ndims = 3 #number of dimensions
Nburnin = 500 # number of burn-in samples
Nsamples = 3500 # number of final posterior samples

argslist = (y, x, err_y, err_x)
p0 = []
for i in range(Nens):
    pi = [
        np.random.uniform(min_,max_), 
        np.random.uniform(min_,max_),
        np.random.uniform((min_scat), (max_scat))
        ]
    p0.append(pi)

if not os.path.isfile(os.getcwd()+'OrthogonalFitting.pkl'):
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

    t0 = time.time()
    sampler.run_mcmc(p0, Nsamples + Nburnin,progress=True);
    t1 = time.time()
    timeemcee = (t1-t0)
    print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    FF = open(f"OrthogonalFitting.pkl", "wb")
    pickle.dump(flat_samples,FF)

FF = open(f"OrthogonalFitting.pkl", "rb")
flat_samples = pickle.load(FF)

#extracting best fit parameters
m_final = np.percentile(flat_samples[:, 0], [16, 50, 84])[1]
c_final = np.percentile(flat_samples[:, 1], [16, 50, 84])[1]
scat_final = np.percentile(flat_samples[:, 2], [16, 50, 84])[1]
Med_value = [m_final,c_final,scat_final]

figure = corner.corner(
    flat_samples,
    figsize=(11,9),
    title_fmt = '.3f',
    levels=(0.68,0.90,0.99), 
    labels=[r"$\alpha$", r"$\beta$", r"$\zeta_{{int}}$"], 
    show_titles=True, 
    label_kwargs={"fontsize": 16},
    title_kwargs={"fontsize": 14},
    color = 'darkorange',alpha=0.1,fill_contours = 1,
)

axis_color='black'
axes = np.array(figure.axes).reshape((ndims, ndims))
for i in range(ndims):
    ax = axes[i, i]
    ax.axvline(Med_value[i], color=axis_color)
for yi in range(ndims):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(Med_value[xi], color=axis_color)
        ax.axhline(Med_value[yi], color=axis_color)
        ax.scatter(Med_value[xi], Med_value[yi], color=axis_color)

plt.legend(
    handles=[mlines.Line2D([], [], color='white',label="STFR")],
    fontsize=16, frameon=False,
    bbox_to_anchor=(1, ndims), loc="upper right"
)
figure.savefig('corner.png',dpi=300)

labels=['slope = ','intercept = ','intrinsic scatter = ',]
line=[]

print('Best-Fit Parameters:')
for i in range(ndims):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(labels[i],round(mcmc[1],3), round(q[0],4), round(-q[1],4))
    line.append(mcmc[1])

x_line=np.linspace(min(x)-0.5,max(x)+0.5)
x_line=np.linspace(min(x)-0.5,max(x)+0.5)
y_line = line[0]*x_line+line[1]
y_fit =[]

#calculating total scatter in teh orthogonal and vertical directions
for n in range(len(x)):
    y_fit.append(np.interp(x[n],x_line,y_line))
scat = np.sqrt(np.median((y-np.array(y_fit))**2))
scat_ort = np.sqrt(np.median((y-np.array(y_fit))**2/(1*line[0]**2+1)))

print('vertical scatter:',round(scat,4))
print('orthogonal scatter:',round(scat_ort,4))

#making a best fit plot
fig= plt.figure( figsize=(8.0,5.5), dpi=100) 
gs = gridspec.GridSpec(4, 4, figure=fig)

#plotting data points
plt.errorbar(x,y, xerr =err_x , yerr=err_y, fmt='o', ms=9, color='black', mfc='royalblue', mew=1, ecolor='gray', alpha=1, capsize=2.0, zorder=1, label='GS23');

#plotting best fit line and 3sigma intrinsic scatter
plt.plot(x_line,y_line, '-', color='darkorange', linewidth=4,zorder=4, label='This work')
plt.plot(x_line,y_line+3*scat, '--', color='orange', linewidth=4,zorder=4, label='3$\sigma$ scatter')
plt.plot(x_line,y_line-3*scat, '--', color='orange', linewidth=4,zorder=4, )

#stylistic changes
plt.ylabel(r'$log(M_{star} \ [\mathrm{M_\odot}])$', fontsize = 16)
plt.xlabel(r'$log(V_c \ [{\rm km/s}])$', fontsize = 16)
ax0 = fig.add_subplot(gs[1:, :3])
plt.ylim(8.0, 12.0)
plt.xlim(1.3, 3.0)
ax1=ax0.twinx()
plt.ylim(8.0, 12.0)
plt.xlim(1.3, 3.0)
ax1.get_yaxis().set_visible(False)
ax1.legend(loc='upper left',fontsize=11)
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)

#saving figure
plt.savefig('bestfit.png',dpi=300)
plt.show()
