#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0,'..')

import matplotlib.pyplot as plt
import numpy as np
import json

from tempo import TempoSys
from tempo import BathInfluence,ProcessTensor,DynamicalProcess
from tempo import Bath
from tempo import System
from tempo import Control
from time import time


#-------- logging ------------------------------------
import logging, sys
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
        filename='mean-field.log',
        filemode='a',
        level=logging.INFO,
        datefmt='%H:%M:%S'
        )
# disable font messages
logging.getLogger('matplotlib.font_manager').disabled = True


#-------- define spin matrices ------------------------

Sx=np.array([[0,0.5],[0.5,0]])
Sy=np.array([[0,-0.5j],[0.5j,0]])
Sz=np.array([[0.5,0],[0,-0.5]])
S11=np.array([[1.0,0],[0,0]])
S21=np.array([[0,0],[1.0,0]])
S12=np.array([[0,1.0],[0,0]])
S22=np.array([[0,0],[0,1.0]])
Sx_up=np.array([[0.5,0.5],[0.5,0.5]])
Id=np.array([[1.0,0],[0,1.0]])

up_state=S11
down_state=S22

#-------- define computation parameters --------------
# All parameters given in meV .

fem_second=1/(6.58*10**2)   # 1 fs = this meV^-1
dt=2.0*fem_second          # timestep
K=301                     # memory cut-off
pp=70                       # precision
tS=0.0*fem_second           # start time
end_step=300               # number of steps (tot. time = dt*end_step)
initial_state=S22           # initial system density matrix


#-------- define system -------------------------------
# Parameters appearing in system Hamiltonian and mean field equations
# Refer to Adv. Quantum Technol. 2019, 2, 1800043 Eq. (1)

w0=1.0                                # two-level system energy splitting
wc=1.0                                # cavity frequency
kap=1/(120*fem_second)                # cavity decay rate
a0=0.0                                # initial field expectation <a>
N=10.0**10                            # number of molecules
gn=10.6*10**-6*np.sqrt(N)             # collective light-matter coupling strength
ga_min=0.0141                         # relaxation rate
interaction_operator=S21              # couples cavity mode to system                      
eta0=np.sqrt(1.0)                     # Gaussian envelope coefficient
sig=20*fem_second                     # pulse width 
ti0=100*fem_second                    # pulse offset



# Define system Hamiltonian for 1 molecule using field expectation alpha=<a>
# May be time-dependent in general
def Hsys(time, alpha):
    molecular_energy = (w0-wc)*Sz
    interaction = gn*(np.conj(alpha)*interaction_operator + alpha*np.conj(interaction_operator.T))
    return molecular_energy + interaction 

# Define equation of motion for alpha=<a> as in d(alpha)/dt = field_eom(alpha, op_expectation)
# Second argument is assigned the expectation value of the interaction_operator defined above
def field_eom(time, alpha, op_expectation): 
    return -(1j*(w0-wc)+kap/2)*alpha - 1j*gn*op_expectation + (eta0/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*((time-ti0)/sig)**2)

# How the field is numerically integrated: alpha_{n+1} = alpha_n + field_increment()
# In TEMPO we have immediate access to the operator expectation at t_n ('current') and t_{n+1}
# ('next') so second-order Runge-Kutte (RK2) is sensible
def field_increment(time, dt, current_alpha, current_op_expectation, next_op_expectation=None):
    # At final iteration we don't know next operator expectation value, so just use Euler
    if next_op_expectation is None:
        return field_eom(time,current_alpha,current_op_expectation)*dt
    # average value now and estimate of value at now+Delta
    rk1 = field_eom(time, current_alpha, current_op_expectation)
    rk2 = field_eom(time +dt, current_alpha + dt*rk1, next_op_expectation)
    return dt*(rk1+rk2)/2
    
# Initialise the system. Mean-field information passed as a dictionary with following fields:
# 'field_increment' (function, 4 args above), 'initial_field' (float) & 'interaction_operator' (2x2)
# N.B. dissipators may be added at this stage
#Define relaxation rate and operator as functions of t, for addition to dissipator list.
gam_min= lambda t:ga_min
diss_op= lambda t:S21

#-------- define bath ------------------------------- 
      
# spectral density cut-off frequency
T=25.68 # bath temperature
# Ohmic spectral density
nuc=150                             # Note couples via Sz

# Initialise for varying a and find resulting <sigz>     
list_a=[]
list_sigz=[] 
for a in range(4,21,4):# system-bath coupling strength
  print("Run  ", a/4)
  jnu = lambda nu: 2*(a/10)*nu*np.exp(-(nu/(nuc))**2)
  print("Computation parameters",K*dt*nuc,dt*gn) 
  list_a.append(a/10)
  bath=Bath(2,Sz,jnu,float(T))  
#-------- define control (unused in this example) ---
  control=Control(dim=2)
  system=System(dim=2, hamiltonian=Hsys, mean_field_info={'field_increment':field_increment,
    'initial_field':a0, 'interaction_operator': interaction_operator}, dissipators=[[gam_min,diss_op]])
# Start logging to mean-field.log
  logging.info('Start TEMPO with mean-field: T={T}, gn {gn}, pp={pp}, K={K}, dt={dt} for '\
        '{end_step} steps.'.format(T=T, gn=gn, pp=pp, K=K, dt=dt, end_step=end_step))

#-------- RUN with tempo_sys ------------------------
  start_time=time()

  temposys=TempoSys(bath,system,control,initial_state,start_time=tS,dt=dt,dkmax=K,precision=pp,
        options={'name':'mean-field','backup_every_nth':0})
  steps0,steps_runtime0=temposys.compute_dynamics(end_step)
#Get times and states from temposys
  t0,states=temposys.get_dynamics()
#Get spin expectations at each point in dynamics (or calculate from states)
  t0,sigz0=temposys.get_expectations(Sz*2,real=True)
#Get corresponding field values <a> from system.field_expectation_array
  fields0=system.field_expectation_array
  run_time=str(round(time()-start_time,2))
  print(sigz0[-1])
  logging.info('Run completed in {}s.'.format(run_time))
  list_sigz.append(sigz0[-1])
#save data
np.array(list_sigz).astype('float').tofile('listsigz.dat')
np.array(list_a).astype('float').tofile('lista.dat')





