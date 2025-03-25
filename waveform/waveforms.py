import pkg_resources

import numpy as np
import scipy
from scipy.integrate import simpson, quad, trapezoid
from scipy.constants import c,G, parsec

Mpc = parsec*10**6
im = complex(0,1)
Msol = 1.9884099*10**30
gamma=np.euler_gamma

class NSBH():
    type = 'Neutron Star Black Hole binary'

    def __init__(self, system_params, z, detector, newtonian=False, step_sizes=np.array([2.23609501e-08, 7.54036265e-10, 4.98486137e-10, 2.72851102e-09, 6.30825767e-09, 2.25155432e-04])):
        ''' Initialisation function. Establishes the source frame properties of the system
            and stores the redshift for transformation at a later point.
            system_params: 1d array of properties, in the following order:
                m_BH: mass of BH in Msol;
                m_NS: mass of second object in Msol;
                r: proper distance to source in Mpc;
                phi, theta: RA, dec of source location on sky;
                iota: angle between line of sight and sysmtem's orb.ang.mom;
                psi: polarisation angle of GWs;
                EOS: the neutron star EOS equation used to model deformability (assumed to be known without error);
            z: redshift of signal;
            newtonian: a flag to say when PN terms (and tidal effects) are being considered;
            step_sizes: used for finite differencing. Defaults to those used in study, but can be varied for optimisation purposes.'''
        
        m_BH, m_NS, r, phi, theta, iota, psi, EOS = system_params
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = np.zeros(6) #finite differencing values
        self.m_BH = m_BH * Msol #source-frame m_BH
        self.m_NS = m_NS * Msol #source-frame m_NS
        self.Mc = (m_BH + m_NS)*((m_BH*m_NS)/((m_BH + m_NS)**2))**(3/5) * Msol #source-frame chirp mass
        self.r = r * Mpc #proper distance
        self.t_c = 6 ## this parameter is arbitrary, set to 6 as this was the value optimisation was performed with
        self.phi_c = 0.1 ## this parameter is arbitrary, set to 0.1 as this was the value optimisation was performed with
        self.z = z
        self.detector = detector
        self.steps = step_sizes
        
        def IFO(detector=detector, theta=theta, phi=phi, psi=psi, iota=iota):
            
            def THETA(source_loc,detector_loc):
                ##function to convert sky (ra,dec) into the relative detector declination
                ra_s, dec_s = source_loc
                ra_d, dec_d = detector_loc
                return np.arccos(np.cos(dec_d)*np.cos(dec_s)*np.cos(ra_s - ra_d) + np.sin(dec_d)*np.sin(dec_s))
            
            def PHI(source_loc, detector_loc):
                ##function to convert sky (ra,dec) into the relative detector right ascension
                ra_s, dec_s = source_loc
                ra_d, dec_d = detector_loc
            
                arm_vec = np.array([1,0])
                rel_source_vec = np.array([ra_s-ra_d,dec_s-dec_d])
                abs = np.sqrt(np.dot(rel_source_vec,rel_source_vec))
                dot = np.dot(arm_vec,rel_source_vec)
                if abs == 0:
                    return 0
                else:
                    return np.arccos(dot/abs)

                
            def ET_IFO(theta, phi, psi, iota):
                def plus(theta, phi, psi):
                    return -np.sqrt(3)/4*((1+np.cos(theta)**2)*np.sin(2*phi)*np.cos(2*psi) + 2*np.cos(theta)*np.cos(2*phi)*np.sin(2*psi))
                def cross(theta, phi, psi):
                    return np.sqrt(3)/4*((1+np.cos(theta)**2)*np.sin(2*phi)*np.sin(2*psi) - 2*np.cos(theta)*np.cos(2*phi)*np.cos(2*psi))
                sum = 0
                for A in [0,1,-1]:
                    sum+= 0.25*(1+np.cos(iota)**2)**2*plus(theta,phi+(A*2*np.pi/3),psi)**2 + np.cos(iota)**2*cross(theta,phi + (A*2*np.pi/3),psi)**2
                return sum

            def CE_IFO(theta, phi, psi, iota):
                def plus(theta, phi, psi):
                    return 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi) -np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
                def cross(theta, phi, psi):
                    return 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi) +np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
                sum = 0.25*(1+np.cos(iota)**2)**2*plus(theta,phi,psi)**2 + np.cos(iota)**2*cross(theta,phi,psi)**2
                return sum

            if detector == 'ET':
                loc = np.array([10.424*np.pi/180, 43.631*np.pi/180])
                angle_func = ET_IFO(theta=THETA([phi,theta],loc), phi=PHI([phi,theta],loc), psi=psi, iota=iota)
            elif detector == 'CE':
                loc = np.array([(-112.825+360)*np.pi/180, 43.827*np.pi/180])
                angle_func = CE_IFO(theta=THETA([phi,theta],loc), phi=PHI([phi,theta],loc), psi=psi, iota=iota)
            else:
                raise Exception('Valid detectors are: ET, CE')
            return np.sqrt(angle_func)
         
        def LambdaFromEOS(mNS, EOS):
            '''calculate the tidal deformation parameter for a NS of given mass and EOS.'''
            if EOS=='APR4':
                LL = (1./mNS**5.) * (-98588.08273630183 + 731070.4271789668 * mNS - \
                                     2.3819745041113514* 10.**6. * mNS**2 + 4.574162110908557* 10.**6. * mNS**3 - \
                                     5.727896605120808 *10.**6. * mNS**4 + 4.88460189278704 * 10.**6. * mNS**5 - \
                                     2.8731016737718047 *10.**6 * mNS**6 + 1.1511822744392804 *10.**6 * mNS**7 - \
                                     300740.113298941 * mNS**8 + 46260.77366514179 * mNS**9 - 3181.874532736834 * mNS**10);
                return LL
            elif EOS=='SLy':
                LL = (1./mNS**5.) * (747844.9903481522 - 5.380764092170365* 10.**6 * mNS + \
                                     1.7290339620432407 * 10.**7 * mNS**2 - 3.2570395127903666 * 10.**7 * mNS**3 + \
                                     3.9838865349514775 * 10.**7 * mNS**4 - 3.306720332013979 * 10.**7 * mNS**5 + \
                                     1.8862143310661916 * 10.**7 * mNS**6 - 7.300895157615867 * 10.**6 * mNS**7 + \
                                     1.8350793251193396 * 10.**6 * mNS**8 - 270447.18458416464 * mNS**9 + \
                                     17744.70795189142 * mNS**10);
                return LL
            elif EOS == 'MPA1':
                LL = (1./mNS**5.) * (-1.9220015300174197 *10.**6 + 1.288760889740576 * 10.**7 * mNS - \
                                     3.8399175751418315 *10.**7 * mNS**2 + 6.7047717916141726 *10.**7 * mNS**3 - \
                                     7.597400565722318 *10.**7 * mNS**4 + 5.8382745852924615 *10.**7 * mNS**5 - \
                                     3.0819302075140323 *10.**7 * mNS**6 + 1.1037899630091587 *10.**7 * mNS**7 - \
                                     2.567500676952564 *10.**6 * mNS**8 + 350346.10805435426 * mNS**9 - \
                                     21302.140619689868 * mNS**10);
                return LL
            elif EOS == 'H4':
                LL = (1./mNS**5.) * (1.3638090418689102 *10.**7 - 9.891883326632795 *10.**7 * mNS + \
                                     3.199639942758125 *10.**8 * mNS**2 - 6.076606034715911 * 10.**8 * mNS**3 + \
                                     7.505479060932645 *10.**8 * mNS**4 - 6.301429088860166 *10.**8 * mNS**5 + \
                                     3.642918052681754 *10.**8 * mNS**6 - 1.4322744726438034 *10.**8 * mNS**7 + \
                                     3.6660715254386336 *10.**7 * mNS**8 - 5.517675770163927 *10.**6 * mNS**9 + \
                                     370879.54916760477 * mNS**10);
                return LL

        self.Q = IFO()
        self.Lambda = LambdaFromEOS(m_NS, EOS)
        self.newtonian = newtonian

    def mass_BH(self):
        ''' Returns the transformed m_BH. '''
        return self.m_BH * (1+self.z)

    def mass_NS(self):
        ''' Returns the transformed m_NS. '''
        return self.m_NS * (1+self.z)

    def symm_mass(self):
        ''' Returns the symmetric mass ratio. '''
        return (1+self.h_eta) * (self.mass_BH()*self.mass_NS())/((self.mass_BH()+self.mass_NS())**2)

    def chirp_mass(self):
        ''' Returns the observer-frame chirp mass. '''
        return (1+self.h_M) * (self.mass_BH()+self.mass_NS())*((self.mass_BH()*self.mass_NS())/((self.mass_BH()+self.mass_NS())**2))**(3/5)

    def dL(self):
        return self.r*(1+self.z)
        
    def total_mass(self):
        ''' Returns the observer-frame total mass, as a function of chirp_mass, symm_mass. '''
        return self.chirp_mass() * self.symm_mass()**(-3/5)

    def prefactor(self):
        ''' Returns the observer-frame amplitude prefactor A. 
            Note that as it is an independent parameter yet a function of chirp_mass. '''
        return (1+self.h_A) * np.sqrt(5/24) * np.pi**(-2/3) * self.Q * (
                ((self.mass_BH()+self.mass_NS())*((self.m_BH*self.m_NS)/((self.m_BH + self.m_NS)**2))**(3/5))**(5/6)) / self.dL() * ((G**5/c**9)**(1/6))

    def t_coalescence(self):
        ''' Returns the observer-frame coalescence time. '''
        return (1+self.h_t) * self.t_c * (1+self.z)

    def phi_coalescence(self):
        ''' Returns the coalescence phase. '''
        return (1+self.h_phi) * self.phi_c
        
    def redshift(self):
        ''' Function for partial differentiating wrt z. '''
        return (1+self.h_z)*self.z
        
    def x(self, frequency):
        ''' Function that creates the dimensionless variable x.
            Defined using source properties, so total_mass() is reduced by (1+z). '''
        return (np.pi * self.total_mass()* frequency * (G/(c**3)))**(2/3)

    def isco_freq(self):
        ''' Calculates the ISCO frequency, the limiting frequency for which the PN approximation is valid.
            Defined using source properties, so total_mass() is reduced by (1+z). '''
        return (6**(3/2) *np.pi * self.chirp_mass() * self.symm_mass()**(-3/5))**(-1) * (c**3/G)
        
    def pp_phase(self, frequency, PN=7):
        '''Calculates the point-particle components of the GW phase under the PN expansion'''
        
        alpha_k = [1,
                   0,
                   3715/756 + 55/9*self.symm_mass(),
                   -16*np.pi,
                   15293365/508032 + 27145/504*self.symm_mass() + 3085/72*self.symm_mass()**2,
                   np.pi*(38645/756 - 65/9*self.symm_mass())*(1+np.log(6**(3/2)*np.pi*self.chirp_mass()*self.symm_mass()**(-3/5)*G/c**3*frequency)),
                   (11583231236531/4694215680 - 640*(np.pi**2)/3 - 6848*gamma/21)+ self.symm_mass()*(-15737765635/3048192+2255/12*np.pi**2) + 76055*(self.symm_mass()**2)/1728 - 127825*(self.symm_mass()**3)/1296 - 6848/63*np.log(64*np.pi*self.chirp_mass()*self.symm_mass()**(-3/5)*frequency*G/c**3),
                   np.pi*(77096675/254016 + 378515*self.symm_mass()/1512 - 74045*(self.symm_mass()**2)/756)
                  ]
        
        sum = 0
        if self.newtonian == True:
            PN = 0
        for k in range(0,PN+1):
            sum += 3/128/self.symm_mass()*alpha_k[k] * self.x(frequency)**((k-5)/2)

        return 2*np.pi*frequency*self.t_coalescence() - self.phi_coalescence() - np.pi/4 + sum
        
    def tidal_phase(self, frequency,tidal_flag=False):
        '''Calculates the tidal component of the GW phase under the PN expansion'''
        if tidal_flag==True:
            return 0
        dimension_factor = 1 #G**4 * c**(-10)
        Sum = 0
        if self.newtonian == True:
            return Sum
        else:
            factor = 3/128*self.symm_mass()**2/(self.chirp_mass()**(5))*dimension_factor

            mm = self.m_NS
            chi = self.mass_NS()/self.chirp_mass()*self.symm_mass()**(3/5)
            xi = -(24/chi) - (264*self.symm_mass()/chi**2)
            zeta =  -15895/28/chi + 4595/28 + 5715*chi/14 - 325*chi**2/7
            Sum = factor*self.Lambda*(mm**5)*((1+self.redshift())**5)*(xi*self.x(frequency)**(5/2) + zeta*self.x(frequency)**(7/2))
            return Sum


    def wave_phase(self, frequency,tidal_flag=False):
        '''Combines the phase elements into a single entity'''
        return self.pp_phase(frequency) + self.tidal_phase(frequency,tidal_flag=tidal_flag)
        
    def waveform(self, frequency,tidal_flag=False):
        '''Computes the GW waveform for each frequency in a given array.
        The only argument of concern is tidal_flag. False indicates that tidal contributions are considered
            whilst True returns a tidal contribution of 0.'''
        return self.prefactor()* frequency**(-7/6) * np.e**(-im * self.wave_phase(frequency,tidal_flag=tidal_flag))

    
    def get_PSD_data(self):
        '''Returns the frequency range and PSD values for a given detector.
        PSD_file: a data file containing two columns: (0) frequency and (1) the PSD at that frequency'''
        PSD_file = pkg_resources.resource_filename('waveform',f'PSDs/{self.detector}_psd.txt')
        PSD_data = [i.strip().split() for i in open(f'{PSD_file}','r').readlines()]
        f = np.array([float(i[0]) for i in PSD_data])
        f_range = f[np.where(f <= self.isco_freq())]
        PSD = np.array([float(i[1]) for i in PSD_data])[np.where(f <= self.isco_freq())]
        spl = scipy.interpolate.UnivariateSpline(f_range,np.log(PSD),k=1,s=0.01)
        xs = np.geomspace(np.min(f_range), np.max(f_range), 10000)
        return xs, np.exp(spl(xs))
        
    def SNR(self):
        '''Computes the SNR of a GW for a given detector.
        PSD_file: the detector's data file. 'get_PSD_data()' will be called using this.
        In computing the waveform, the self.waveform() function takes the range of values for which the PSD detector is evaluated.
        As the PSD is not a continuous function, the SNR is calculated by using Simpson integration (summing over the function for given frequencies).'''
        f, psd = self.get_PSD_data()
        SNR = np.sqrt(4*trapezoid(y=(np.abs(self.waveform(f))**2/psd), x=f))
        return SNR

    def get_central_diff(self, diff_param, frequency):
        def get_difference_param(diff_param):
            '''Returns the parameter that is being differentiated without finite difference perturbation. Necessary as a consequence of how 
                I have implemented numerical differentiation.'''
            index = np.where(np.abs(diff_param) > 0)[0][0]
            difference_params = [self.prefactor(), self.chirp_mass(),
                                 self.t_coalescence(), self.phi_coalescence(),
                                 self.symm_mass(), self.redshift()]
            return difference_params[index]

        param = get_difference_param(diff_param=diff_param)
        
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = diff_param
        difference_plus = self.waveform(frequency)
        
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = -diff_param
        difference_minus = self.waveform(frequency)
        
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = np.zeros(6)
        partial = (difference_plus - difference_minus)/(2*np.sum(diff_param)*param)
        
        return partial
        
        
    def get_lower_diff(self, diff_param, frequency):
        def get_difference_param(diff_param):
            '''Returns the parameter that is being differentiated without finite difference perturbation. Necessary as a consequence of how 
                I have implemented numerical differentiation.'''
            index = np.where(np.abs(diff_param) > 0)[0][0]
            difference_params = [self.prefactor(), self.chirp_mass(),
                                 self.t_coalescence(), self.phi_coalescence(),
                                 self.symm_mass(), self.redshift()]
            return difference_params[index]

        param = get_difference_param(diff_param=diff_param)
        
        regular = self.waveform(frequency)
        
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = -diff_param
        difference_minus = self.waveform(frequency)

        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = np.zeros(6)
        partial = (regular - difference_minus)/(np.sum(diff_param)*param)
        return partial   


    def get_FIM(self):
        '''Function to compute and return the Fisher information matrix.
        h: the finite difference vector for computation as found through optimisation for a [5.08,1] NSBH.'''
        def inner_product(fnc_i, fnc_j, PSD, frequency):
            integrand = 2 * (np.conjugate(fnc_i)*fnc_j + fnc_i*np.conjugate(fnc_j))/PSD
            inner_prod = trapezoid(y=integrand, x=frequency)
            return inner_prod
            
        h1,h2,h3,h4,h5,h6 = self.steps
        H_A   = np.array([h1,0,0,0,0,0])
        H_M   = np.array([0,h2,0,0,0,0])
        H_t   = np.array([0,0,h3,0,0,0])
        H_phi = np.array([0,0,0,h4,0,0])
        H_eta = np.array([0,0,0,0,h5,0])
        H_z   = np.array([0,0,0,0,0,h6])      
        
        f,psd_data = self.get_PSD_data()
        
        ## FIM setup
        fisher_matrix = []
        param_row = [self.prefactor()*self.get_central_diff(H_A,f),self.chirp_mass()*self.get_central_diff(H_M,f),
                     self.get_central_diff(H_t,f),self.get_central_diff(H_phi,f),self.get_lower_diff(H_eta,f),self.get_central_diff(H_z,f)] 
    
        ## Looping over row elements to create FIM
        fisher_matrix = []
        for i in param_row:
            row = []
            for j in param_row:
                row.append(inner_product(fnc_i=i, fnc_j=j, PSD=psd_data, frequency=f))
            fisher_matrix.append(row) 
        return np.real(np.array(fisher_matrix))

    def tidal_missmatch(self,cycles=False):
        '''Computes the missmatch of a signal when tidal contributions are and aren't considered.
        cycles: if False, return the inner product of standard waveform against tidally omitted waveform.
                If True, calculate the accumulated phase of the signal with and without tidal contributions
                and return the difference in observed cycles (by reducing by 2pi).'''
        f, psd = self.get_PSD_data()
        fnc_i = self.waveform(f,tidal_flag=False)
        fnc_j = self.waveform(f,tidal_flag=True)
        if cycles==False:
            integrand = 2 * (np.conjugate(fnc_i)*fnc_j + fnc_i*np.conjugate(fnc_j))/psd
            return np.real(trapezoid(y=integrand,x=f)/self.SNR()**2)
        else:
            return (trapezoid(self.wave_phase(f,tidal_flag=True),f)-trapezoid(self.wave_phase(f,tidal_flag=False),f))/2/np.pi


class BNS():
    type = 'Binary Neutron Star'

    def __init__(self, system_params, z, detector, newtonian=False):
        ''' Initialisation function. Establishes the source frame properties of the system
            and stores the redshift for transformation at a later point.
            system_params: 1d array of properties, in the following order:
                m1: mass of first object in Msol;
                m2: mass of second object in Msol;
                r: proper distance to source in Mpc;
                phi, theta: RA, dec of source location on sky;
                iota: angle between line of sight and sysmtem's orb.ang.mom;
                psi: polarisation angle of GWs;
                t_c: coordinate time of coalescence;
                phi_c: phase at coalescence;
                EOS: the neutron star EOS equation used to model deformability (assumed to be known without error);
            z: redshift of signal;
            H: 1d array of finite difference values used for partial derivatives, in the order of
                [logA, log(chirp mass), eta, t_c, phi_c]
                note that there is no entry for z, as finite differencing will be done by creating another
                class instance of z = (1+h_z)*z;
            newtonian: a flag to say when PN terms (and tidal effects) are being considered. '''
        self.H = H
        m1, m2, r, phi, theta, iota, psi, t_c, phi_c, EOS = system_params
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = H #finite differencing values
        self.m1 = m1 * Msol #source-frame m1
        self.m2 = m2 * Msol #source-frame m2
        self.Mc = (m1 + m2)*((m1*m2)/((m1 + m2)**2))**(3/5) * Msol #source-frame chirp mass
        self.r = r * Mpc #proper distance
        self.t_c = t_c
        self.phi_c = phi_c
        self.z = z
        self.detector = detector

        def IFO(detector=detector, theta=theta, phi=phi, psi=psi, iota=iota):
            
            def ET_IFO(theta, phi, psi, iota):
                #assume coords are the tripoint of Belgium, Netherlands and Germany 
                def plus(theta, phi, psi):
                    return -np.sqrt(3)/4*((1+np.cos(theta)**2)*np.sin(2*phi)*np.cos(2*psi) + 2*np.cos(theta)*np.cos(2*phi)*np.sin(2*psi))
                def cross(theta, phi, psi):
                    return np.sqrt(3)/4*((1+np.cos(theta)**2)*np.sin(2*phi)*np.sin(2*psi) - 2*np.cos(theta)*np.cos(2*phi)*np.cos(2*psi))
                sum = 0
                for A in [0,1,-1]:
                    sum+= 0.25*(1+np.cos(iota)**2)**2*plus(theta,phi+(A*2*np.pi/3),psi)**2 + np.cos(iota)**2*cross(theta,phi + (A*2*np.pi/3),psi)**2
                return sum

            def CE_IFO(theta, phi, psi, iota):
                #assume same coords as LIGO Livinston: 30°33'46.3"N 90°46'29.1"W
                def plus(theta, phi, psi):
                    return 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi) -np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
                def cross(theta, phi, psi):
                    return 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi) -np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
                sum = 0.25*(1+np.cos(iota)**2)**2*plus(theta,phi,psi)**2 + np.cos(iota)**2*cross(theta,phi,psi)**2
                return sum

            if detector == 'ET':
                angle_func = ET_IFO(theta=theta, phi=phi, psi=psi, iota=iota)
            elif detector == 'CE':
                angle_func = CE_IFO(theta=theta, phi=phi, psi=psi, iota=iota)
            else:
                raise Exception('Valid detectors are: ET, CE')
            return np.sqrt(angle_func)
         
        def LambdaFromEOS(mNS, EOS):
            if EOS=='APR4':
                LL = (1./mNS**5.) * (-98588.08273630183 + 731070.4271789668 * mNS - \
                                     2.3819745041113514* 10.**6. * mNS**2 + 4.574162110908557* 10.**6. * mNS**3 - \
                                     5.727896605120808 *10.**6. * mNS**4 + 4.88460189278704 * 10.**6. * mNS**5 - \
                                     2.8731016737718047 *10.**6 * mNS**6 + 1.1511822744392804 *10.**6 * mNS**7 - \
                                     300740.113298941 * mNS**8 + 46260.77366514179 * mNS**9 - 3181.874532736834 * mNS**10);
                return LL
            elif EOS=='SLy':
                LL = (1./mNS**5.) * (747844.9903481522 - 5.380764092170365* 10.**6 * mNS + \
                                     1.7290339620432407 * 10.**7 * mNS**2 - 3.2570395127903666 * 10.**7 * mNS**3 + \
                                     3.9838865349514775 * 10.**7 * mNS**4 - 3.306720332013979 * 10.**7 * mNS**5 + \
                                     1.8862143310661916 * 10.**7 * mNS**6 - 7.300895157615867 * 10.**6 * mNS**7 + \
                                     1.8350793251193396 * 10.**6 * mNS**8 - 270447.18458416464 * mNS**9 + \
                                     17744.70795189142 * mNS**10);
                return LL
            elif EOS == 'MPA1':
                LL = (1./mNS**5.) * (-1.9220015300174197 *10.**6 + 1.288760889740576 * 10.**7 * mNS - \
                                     3.8399175751418315 *10.**7 * mNS**2 + 6.7047717916141726 *10.**7 * mNS**3 - \
                                     7.597400565722318 *10.**7 * mNS**4 + 5.8382745852924615 *10.**7 * mNS**5 - \
                                     3.0819302075140323 *10.**7 * mNS**6 + 1.1037899630091587 *10.**7 * mNS**7 - \
                                     2.567500676952564 *10.**6 * mNS**8 + 350346.10805435426 * mNS**9 - \
                                     21302.140619689868 * mNS**10);
                return LL
            elif EOS == 'H4':
                LL = (1./mNS**5.) * (1.3638090418689102 *10.**7 - 9.891883326632795 *10.**7 * mNS + \
                                     3.199639942758125 *10.**8 * mNS**2 - 6.076606034715911 * 10.**8 * mNS**3 + \
                                     7.505479060932645 *10.**8 * mNS**4 - 6.301429088860166 *10.**8 * mNS**5 + \
                                     3.642918052681754 *10.**8 * mNS**6 - 1.4322744726438034 *10.**8 * mNS**7 + \
                                     3.6660715254386336 *10.**7 * mNS**8 - 5.517675770163927 *10.**6 * mNS**9 + \
                                     370879.54916760477 * mNS**10);
                return LL

        # self.Q = nuisance_params(theta, phi, psi, iota)
        self.Q = IFO()
        self.Lambda1 = LambdaFromEOS(m1, EOS)
        self.Lambda2 = LambdaFromEOS(m2, EOS)
        self.newtonian = newtonian

    def mass_1(self):
        ''' Returns the transformed m1. '''
        return self.m1 * (1+self.z)

    def mass_2(self):
        ''' Returns the transformed m2. '''
        return self.m2 * (1+self.z)

    def symm_mass(self):
        ''' Returns the symmetric mass ratio. '''
        return (1+self.h_eta) * (self.mass_1()*self.mass_2())/((self.mass_1()+self.mass_2())**2)

    def chirp_mass(self):
        ''' Returns the observer-frame chirp mass. '''
        return (1+self.h_M) * (self.mass_1()+self.mass_2())*((self.mass_1()*self.mass_2())/((self.mass_1()+self.mass_2())**2))**(3/5)

    def dL(self):
        return self.r*(1+self.z)
        
    def total_mass(self):
        ''' Returns the observer-frame total mass, as a function of chirp_mass, symm_mass. '''
        return self.chirp_mass() * self.symm_mass()**(-3/5)

    def prefactor(self):
        ''' Returns the observer-frame amplitude prefactor A. 
            Note that as it is an independent parameter yet a function of chirp_mass. '''
        return (1+self.h_A) * np.sqrt(5/24) * np.pi**(-2/3) * self.Q * (
                ((self.mass_1()+self.mass_2())*((self.m1*self.m2)/((self.m1 + self.m2)**2))**(3/5))**(5/6)) / self.dL() * ((G**5/c**9)**(1/6))
        # return (1+self.h_A) * np.sqrt(5/24) * np.pi**(-2/3) * self.Q * (self.Mc**(5/6))/self.r * ((G**5/c**9)**(1/6))

    def t_coalescence(self):
        ''' Returns the observer-frame coalescence time. '''
        return (1+self.h_t) * self.t_c * (1+self.z)

    def phi_coalescence(self):
        ''' Returns the coalescence phase. '''
        return (1+self.h_phi) * self.phi_c
        
    def redshift(self):
        ''' Function for partial differentiating wrt z. '''
        return (1+self.h_z)*self.z
        
    def x(self, frequency):
        ''' Function that creates the dimensionless variable x.
            Defined using source properties, so total_mass() is reduced by (1+z). '''
        return (np.pi * self.total_mass()* frequency * (G/(c**3)))**(2/3)

    def isco_freq(self):
        ''' Calculates the ISCO frequency, the limiting frequency for which the PN approximation is valid.
            Defined using source properties, so total_mass() is reduced by (1+z). '''
        return (6**(3/2) *np.pi * self.chirp_mass() * self.symm_mass()**(-3/5))**(-1) * (c**3/G)
        
    def pp_phase(self, frequency, PN=7):
        '''Calculates the point-particle components of the GW phase under the PN expansion'''
        
        alpha_k = [1,
                   0,
                   3715/756 + 55/9*self.symm_mass(),
                   -16*np.pi,
                   15293365/508032 + 27145/504*self.symm_mass() + 3085/72*self.symm_mass()**2,
                   np.pi*(38645/756 - 65/9*self.symm_mass())*(1+np.log(6**(3/2)*np.pi*self.chirp_mass()*self.symm_mass()**(-3/5)*G/c**3*frequency)),
                   (11583231236531/4694215680 - 640*(np.pi**2)/3 - 6848*gamma/21)+ self.symm_mass()*(-15737765635/3048192+2255/12*np.pi**2) + 76055*(self.symm_mass()**2)/1728 - 127825*(self.symm_mass()**3)/1296 - 6848/63*np.log(64*np.pi*self.chirp_mass()*self.symm_mass()**(-3/5)*frequency*G/c**3),
                   np.pi*(77096675/254016 + 378515*self.symm_mass()/1512 - 74045*(self.symm_mass()**2)/756)
                  ]
        
        sum = 0
        if self.newtonian == True:
            PN = 0
        for k in range(0,PN+1):
            sum += 3/128/self.symm_mass()*alpha_k[k] * self.x(frequency)**((k-5)/2)

        return 2*np.pi*frequency*self.t_coalescence() - self.phi_coalescence() - np.pi/4 + sum
        
    def tidal_phase(self, frequency,tidal_flag=False):
        '''Calculates the tidal component of the GW phase under the PN expansion.'''
        if tidal_flag ==True:
            return 0
        dimension_factor = 1 #G**4 * c**(-10)
        sum = 0
        if self.newtonian == True:
            return sum
        else:
            factor = 3/128*self.symm_mass()**2/(self.chirp_mass()**(5))*dimension_factor
            LL = np.array([self.Lambda1, self.Lambda2])
            mm = np.array([self.m1, self.m2])
            chi = np.array([self.mass_1(), self.mass_2()])/self.chirp_mass()*self.symm_mass()**(3/5)
            for n in [0,1]:
                xi = -(24/chi[n]) - (264*self.symm_mass()/chi[n]**2)
                zeta =  -15895/28/chi[n] + 4595/28 + 5715*chi[n]/14 - 325*chi[n]**2/7
                sum += factor*LL[n]*(mm[n]**5)*((1+self.redshift())**5)*(xi*self.x(frequency)**(5/2) + zeta*self.x(frequency)**(7/2))
            return sum


    def wave_phase(self, frequency,tidal_flag=False):
        '''Combines the phase elements into a single entity'''
        return self.pp_phase(frequency) + self.tidal_phase(frequency,tidal_flag=tidal_flag)
        
    def waveform(self, frequency,tidal_flag=False):
        '''Computes the GW waveform for each frequency in a given array'''
        return self.prefactor()* frequency**(-7/6) * np.e**(-im * self.wave_phase(frequency,tidal_flag=tidal_flag)) #*(1+self.redshift())**(-1/6)

    def get_PSD_data(self):
        '''Returns the frequency range and PSD values for a given detector.
        PSD_file: a data file containing two columns: (0) frequency and (1) the PSD at that frequency'''
        PSD_file = pkg_resources.resource_filename('waveform',f'PSDs/{self.detector}_psd.txt')
        PSD_data = [i.strip().split() for i in open(f'{PSD_file}','r').readlines()]
        f = np.array([float(i[0]) for i in PSD_data])
        f_range = f[np.where(f <= self.isco_freq())]
        PSD = np.array([float(i[1]) for i in PSD_data])[np.where(f <= self.isco_freq())]
        spl = scipy.interpolate.UnivariateSpline(f_range,np.log(PSD),k=1,s=0.01)
        xs = np.geomspace(np.min(f_range), np.max(f_range), 10000)
        return xs, np.exp(spl(xs))
        
    def SNR(self):
        '''Computes the SNR of a GW for a given detector.
        PSD_file: the detector's data file. 'get_PSD_data()' will be called using this.
        In computing the waveform, the self.waveform() function takes the range of values for which the PSD detector is evaluated.
        As the PSD is not a continuous function, the SNR is calculated by using Simpson integration (summing over the function for given frequencies).'''
        f, psd = self.get_PSD_data()
        SNR = np.sqrt(4*trapezoid(y=np.abs(self.waveform(f))**2/psd, x=f))
        return SNR

    def get_central_diff(self, diff_param, frequency):
        def get_difference_param(diff_param):
            '''Returns the parameter that is being differentiated without finite difference perturbation. Necessary as a consequence of how 
                I have implemented numerical differentiation (see mathtools).'''
            index = np.where(np.abs(diff_param) > 0)[0][0]
            difference_params = [self.prefactor()/(1+sum(diff_param)), self.chirp_mass()/(1+sum(diff_param)),
                                 self.t_coalescence()/(1+sum(diff_param)), self.phi_coalescence()/(1+sum(diff_param)),
                                 self.symm_mass()/(1+sum(diff_param)), self.redshift()/(1+sum(diff_param)) ]
            return difference_params[index]
    
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = diff_param
        param = get_difference_param(diff_param=diff_param)
        difference_plus = self.waveform(frequency)
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = -diff_param
        difference_minus = self.waveform(frequency)
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = np.zeros(6)
        partial = (difference_plus - difference_minus)/(2*np.sum(diff_param)*param)
        return partial
        
        
    def get_lower_diff(self, diff_param, frequency):
        def get_difference_param(diff_param):
            '''Returns the parameter that is being differentiated without finite difference perturbation. Necessary as a consequence of how 
                I have implemented numerical differentiation (see mathtools).'''
            index = np.where(np.abs(diff_param) > 0)[0][0]
            difference_params = [self.prefactor()/(1+sum(diff_param)), self.chirp_mass()/(1+sum(diff_param)),
                                 self.t_coalescence()/(1+sum(diff_param)), self.phi_coalescence()/(1+sum(diff_param)),
                                 self.symm_mass()/(1+sum(diff_param)), self.redshift()/(1+sum(diff_param)) ]
            return difference_params[index]
        regular = self.waveform(frequency)
        self.h_A, self.h_M, self.h_t, self.h_phi, self.h_eta, self.h_z = diff_param
        param = get_difference_param(diff_param=diff_param)
        difference_minus = self.waveform(frequency)
        partial = (regular - difference_minus)/(np.sum(diff_param)*param)
        return partial   


    def get_FIM(self, h = np.array([1.85048917e-10, 6.52941123e-10, 1.80585441e-10, 1.44727541e-9, 4.17432481e-10, 1e-7])):
        '''Function to compute and return Fisher information matrix
        h: the finite difference vector for computation as found for a [1.4,1.4] BNS.'''
        def inner_product(fnc_i, fnc_j, PSD, frequency):
            integrand = 2 * (np.conjugate(fnc_i)*fnc_j + fnc_i*np.conjugate(fnc_j))/PSD
            inner_prod = trapezoid(y=integrand, x=frequency)
            return inner_prod
            
        h1,h2,h3,h4,h5,h6 = h
        H_A   = np.array([h1,0,0,0,0,0])
        H_M   = np.array([0,h2,0,0,0,0])
        H_t   = np.array([0,0,h3,0,0,0])
        H_phi = np.array([0,0,0,h4,0,0])
        H_eta = np.array([0,0,0,0,h5,0])
        H_z   = np.array([0,0,0,0,0,h6])      
        
        f,psd_data = self.get_PSD_data()
        
        ## FIM setup
        fisher_matrix = []
        param_row = [self.prefactor()*self.get_central_diff(H_A,f),self.chirp_mass()*self.get_central_diff(H_M,f),
                     self.get_central_diff(H_t,f),self.get_central_diff(H_phi,f),self.get_lower_diff(H_eta,f),self.get_central_diff(H_z,f)] 
    
        ## Looping over row elements to create FIM
        fisher_matrix = []
        for i in param_row:
            row = []
            for j in param_row:
                row.append(inner_product(fnc_i=i, fnc_j=j, PSD=psd_data, frequency=f))
            fisher_matrix.append(row) 
        return np.real(np.array(fisher_matrix))
        
    def tidal_missmatch(self):
        '''Computes the missmatch of a signal when tidal contributions are and aren't considered.
        cycles: if False, return the inner product of standard waveform against tidally omitted waveform.
                If True, calculate the accumulated phase of the signal with and without tidal contributions
                and return the difference in observed cycles (by reducing by 2pi).'''
        f, psd = self.get_PSD_data()
        fnc_i = self.waveform(f,tidal_flag=False)
        fnc_j = self.waveform(f,tidal_flag=True)
        integrand = 2 * (np.conjugate(fnc_i)*fnc_j + fnc_i*np.conjugate(fnc_j))/psd
        return np.real(trapezoid(y=integrand,x=f)/self.SNR()**2)