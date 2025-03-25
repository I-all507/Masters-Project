import numpy as np

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import trapezoid

from .waveforms import NSBH, BNS
from scipy.constants import c,G
gamma = np.euler_gamma
im = complex(0,1)


def proper_distance(zed):
    '''Computes and returns the proper distance, assuming a Plank18 cosmology.'''
    return float(cosmo.luminosity_distance(zed)/(1+zed)/u.Mpc)

def inner_product(fnc_i, fnc_j, PSD, frequency):
    '''Compute the noise--weighted inner product of two functions for a given PSD.
    fnc_i, fnc_j: the two functions to be integrated over;
    PSD: the PSD data for the detector being considered (see waveforms.NSBH.get_PSD_data);
    frequency: frequency range over which to integrate.'''
    integrand = 2 * (np.conjugate(fnc_i)*fnc_j + fnc_i*np.conjugate(fnc_j))/PSD
    inner_prod = trapezoid(y=integrand, x=frequency)
    return inner_prod

def get_IFIM_errors(FIM):
    '''Computes and returns the inverse of a given Fisher information matrix.
    FIM: NxN Fisher information matrix.'''
    IFIM = np.linalg.inv(FIM)
    errors = np.sqrt(np.diag(IFIM))
    return IFIM, errors

def get_analyt(params,z,detector):
    '''Computes the analytical IFIM and parameter uncertainties for a given parametrised NSBH.
    params: 1d array containing parameters, ordered identically to the NSBH class 'system_params'. '''

    ## I'm sorry, this is very messy!
    
    test = NSBH(system_params=params, z=z,detector=detector)
    frequency, psd_data = test.get_PSD_data() 
       
    alpha_k = [1,
               0,
               3715/756 + 55/9*test.symm_mass(),
               -16*np.pi,
               15293365/508032 + 27145/504*test.symm_mass() + 3085/72*test.symm_mass()**2,
               np.pi*(38645/756 - 65/9*test.symm_mass())*(1+np.log(6**(3/2)*np.pi*test.chirp_mass()*test.symm_mass()**(-3/5)*G/c**3*frequency)),
               (11583231236531/4694215680 - 640*(np.pi**2)/3 - 6848*gamma/21)+ test.symm_mass()*(-15737765635/3048192+2255/12*np.pi**2) + 76055*(test.symm_mass()**2)/1728 - 127825*(test.symm_mass()**3)/1296 - 6848/63*np.log(64*np.pi*test.chirp_mass()*test.symm_mass()**(-3/5)*frequency*G/c**3),
               np.pi*(77096675/254016 + 378515*test.symm_mass()/1512 - 74045*(test.symm_mass()**2)/756)
              ]
    
    ## dlmM alpha_k derivatives
    beta_k = [0,0,0,0,0,np.pi*(38645/756 - 65*test.symm_mass()/9),-6848/63,0]
    
    ## d_eta alpha_k derivatives 
    gamma_k = [0,0,55/9,0,
               27145/504 + 3085/36*test.symm_mass(),
               -65/9*np.pi*(1+np.log(6**(3/2)*np.pi*test.chirp_mass()*test.symm_mass()**(-3/5)*G/c**3*frequency)) - 3*np.pi/5/test.symm_mass()*(38645/756 - 65/9*test.symm_mass()),
               (-15737765635/3048192 + 2255/12*np.pi**2) + 76055/864*test.symm_mass() - 383475/1296*test.symm_mass()**2 +6848/105/test.symm_mass(),
               np.pi*(378515/1512 - 74045/378*test.symm_mass())]
    
    
    
    dlnM_coeff = 0
    for k in range(len(beta_k)):
        dlnM_coeff += 3/128/test.symm_mass()*((k-5)/3*alpha_k[k]+beta_k[k])*test.x(frequency)**((k-5)/2)
    LL = np.array([test.Lambda, test.Lambda])
    mm = np.array([test.m_BH, test.m_NS])
    chi = np.array([test.mass_BH(), test.mass_NS()])/test.chirp_mass()*test.symm_mass()**(3/5)
    for a in [1]:
        xi = -(24/chi[a]) - (264*test.symm_mass()/chi[a]**2)
        zeta =  -15895/28/chi[a] + 4595/28 + 5715*chi[a]/14 - 325*chi[a]**2/7
        xi_m = -24/chi[a] - 528/chi[a]**2*test.symm_mass()
        zeta_m = -15895/28/chi[a] - 5715/14*chi[a] + 650/7*chi[a]**2
        dlnM_coeff += 3/128*test.symm_mass()**2*LL[a]*(mm[a]/test.chirp_mass())**5*(1+test.z)**5*((xi_m - 10/3*xi)*test.x(frequency)**(5/2) + (zeta_m - 8/3*zeta)*test.x(frequency)**(7/2))
    dlnM_coeff = dlnM_coeff*-im
    
    deta_coeff = 0
    for k in range(len(gamma_k)):
        deta_coeff += 3/128/test.symm_mass()*(gamma_k[k]-k*alpha_k[k]/5/test.symm_mass())*(test.x(frequency)**((k-5)/2))
    for a in [1]:
        xi = -(24/chi[a]) - (264*test.symm_mass()/chi[a])
        zeta =  -15895/28/chi[a] + 4595/28 + 5715*chi[a]/14 - 325*chi[a]**2/7
        xi_eta = 72/5/test.symm_mass()/chi[a] + 1584/5/chi[a]**2 - 264/chi[a]**2
        zeta_eta = 9537/28/test.symm_mass()/chi[a] + 3429/14/test.symm_mass()*chi[a] - 390/7/test.symm_mass()*chi[a]**2
        deta_coeff += 3/128*test.symm_mass()**2*LL[a]*(mm[a]/test.chirp_mass())**5*(1+test.z)**5*((xi_eta + 1/test.symm_mass()*xi)*test.x(frequency)**(5/2) + (zeta_eta + 3/5/test.symm_mass()*zeta)*test.x(frequency)**(7/2))
    deta_coeff = deta_coeff * -im
    
    dz_coeff = -im*5/(1+test.z)*test.tidal_phase(frequency)
    
    
    
    dlnA = test.waveform(frequency)
    dlnM = test.waveform(frequency) * dlnM_coeff
    dt_c = -2*np.pi*im*frequency*test.waveform(frequency)
    dphi = im * test.waveform(frequency)
    deta = deta_coeff*test.waveform(frequency)
    dz = dz_coeff*test.waveform(frequency)
    
    param_row = [dlnA, dlnM, dt_c, dphi, deta,dz]
    
    fisher_matrix = []
    for i in param_row:
        row = []
        for j in param_row:
            row.append(inner_product(fnc_i=i, fnc_j=j, PSD=psd_data, frequency=frequency))
        fisher_matrix.append(row)
    inverse_fisher_matrix = np.linalg.inv(np.array(fisher_matrix))
    param_errors = np.real(np.sqrt(np.diag(inverse_fisher_matrix)))
    return(inverse_fisher_matrix,param_errors)


def optimiser(analytical_errors, params, detector, h0, z, h_limit=1e-12, score_limit=1e-5, factor=1,maxiter=100,itercount=0,yapper=False):
    '''Function used to calibrate the step sizes for finite differencing. Works like a Hamilton walker, where the point in ''step space''
       walks in the direction that minimises the distance between the IFIM errors using those step sizes and the analytical results.
    analytical_errors: 1d array containing the analytical uncertainties. Corresponds to the second element returned from 'get_analyt()';
    params: the 1d array that parametrises instances of the NSBH class;
    detector: the detector in question;
    h0: injected location in step space;
    z: the redshift of the signal;
    h_limit: the smallest allowed step size. Use this with caution, as the walker likes to minimise the step sizes for the prefactor and chirp mass,
        but this actually causes the results to become rather unstable when used for a different signal. 1e-12 should be treated as the absolute minimum;
    score_limit: the distance between analytical and numerical uncertainties which causes the optimiser to cease searching step-space. Note that decreasing the score
        can cause the optimisation time to increase drastically;
    factor: used for fine-tuning the random walk length;
    maxiter: how many random walks may be taken before the optimisation stops. Usually ceases long before this value is reached (~10-20 in bad cases);
    itercount: keeps track of what iteration the optimiser is on;
    yapper: set to True for additional comments on the optimisation process.
    '''
        
       
    def get_vect_IFIM(params,z,h,detector=detector):
        ## score is measured using the logged IFIM results. This function computes them.
        system = NSBH(params,z=z,detector=detector,step_sizes=h)
        fim = system.get_FIM()
        _,IFIM_vector=get_IFIM_errors(fim)
        return np.log(IFIM_vector) #returns logged IFIM values
    
    log_node = np.log(analytical_errors) #logging the analytical results, referred to as the node.

    if maxiter == 0: #exit case
        print('Maximum iterations reached.')
        return np.exp(get_vect_IFIM(params, h=h0, z=z)), h0
        
    else:
        IFIM_vector = get_vect_IFIM(params,h=h0,z=z) # get the log IFIM of the current position in step space.

        def get_score(vector):
            ## score defined as the dot-product of the difference between analytical and numerical logged uncertainties.
            def difference(point, log_node=log_node):
                return np.abs(log_node-point)
            return np.dot(difference(point=vector),difference(point=vector))
            
        score = get_score(IFIM_vector) # get the score of the current position in step space.
        
        if score <= score_limit: # exit case for when desired accuracy is met.
            print('Sufficient accuracy reached. Converged on h = {} after {} iterations.'.format(h0,itercount))
            return np.exp(get_vect_IFIM(params, h=h0,z=z)), h0
        
        tick = 300 #maximum number of attempts to find a better value of h, will exit function if tick <0.
        while tick > 0:
            tick -= 1

            #randomly selecting new h vector. The factor defines how small of a purturbation is applied to h0.
            def h_purturber(h,factor=factor,h_limit=h_limit): 
                ## takes a random walk in step space, length determined by the factor
                temp = h*(-1+(2*np.random.rand(len(h))*factor))
                upper = h + temp
                return np.piecewise(upper, [upper < h_limit, upper >= h_limit], [h_limit, lambda upper: upper])


            #computing new scores using varied h values
            h_prime = h_purturber(h0,h_limit=h_limit)
            score_prime = get_score(get_vect_IFIM(params, h=h_prime,z=z))

            #score comparisons. If either of new h values has a lowest score, rerun the function with new h value
            #otherwise, the purturbation factor will become more and more precise in an attempt to find a better h.
            if (score_prime < score):
                if yapper == True:
                    print('Current score: {}, new score: {}, new h: {}'.format(score,score_prime,h_prime))
                return optimiser(analytical_errors, params=params, score_limit=score_limit,
                                      detector=detector, h0=h_prime, z=z,
                                      h_limit=h_limit, factor=1, maxiter=maxiter-1,
                                      itercount = itercount + 1,yapper=yapper)
            else:
                factor = factor/1.01 # NOTE: factor reduces with each failed attempt. This is in case the minimum is in a small hole and keeps getting leaped over.

        #if no better value of h was found, function will return current h value, and conclude that this is the optimal h.
        print('Converged on h = {} after {} iterations.'.format(h0,itercount))
        
        return np.exp(get_vect_IFIM(params, h=h0, z=z)), h0 #returns re-exponentiated IFIM elements