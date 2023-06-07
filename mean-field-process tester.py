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

dt=0.05/2   # timestep
K=60*2       # memory cut-off
pp=60       # precision
tS=0.0      # start time
end_step=100*2# number of steps (tot. time = dt*end_step)
initial_state=Sx_up # initial system density matrix


#-------- define system -------------------------------
# Parameters appearing in system Hamiltonian and mean field equations
# Refer to Adv. Quantum Technol. 2019, 2, 1800043 Eq. (1)
w0=2.0  # two-level system energy splitting
wc=2.0  # cavity frequency
kap=1.0# cavity decay rate
a0=1.0  # initial field expectation <a>
gn=1.0  # collective light-matter coupling strength
interaction_operator=Sx # couples cavity mode to system

# Define system Hamiltonian for 1 molecule using field expectation alpha=<a>
# May be time-dependent in general
def Hsys(time, alpha):
    molecular_energy = w0*Sz # (unnecessary) 
    interaction = gn*2*np.real(alpha)*interaction_operator
    return  molecular_energy + interaction

# Define equation of motion for alpha=<a> as in d(alpha)/dt = field_eom(alpha, op_expectation)
# Second argument is assigned the expectation value of the interaction_operator defined above
def field_eom(alpha, op_expectation): 
    return -(1j*wc+kap)*alpha - 1j*gn*op_expectation

# How the field is numerically integrated: alpha_{n+1} = alpha_n + field_increment()
# In TEMPO we have immediate access to the operator expectation at t_n ('current') and t_{n+1}
# ('next') so second-order Runge-Kutte (RK2) is sensible
# N.B. time dependent in general
def field_increment(time, dt, current_alpha, current_op_expectation, next_op_expectation=None):
    # At final iteration we don't know next operator expectation value, so just use Euler
    if next_op_expectation is None:
        return field_eom(current_alpha,current_op_expectation)*dt
    # average value now and estimate of value at now+Delta
    rk1 = field_eom(current_alpha, current_op_expectation)
    rk2 = field_eom(current_alpha + dt*rk1, next_op_expectation)
    return dt*(rk1+rk2)/2

# Initialise the system. Mean-field information passed as a dictionary with following fields:
# 'field_increment' (function, 4 args above), 'initial_field' (float) & 'interaction_operator' (2x2)
# N.B. dissipators may be added at this stage
system=System(dim=2, hamiltonian=Hsys, mean_field_info={'field_increment':field_increment,
    'initial_field':a0, 'interaction_operator': interaction_operator})

#-------- define bath -------------------------------

a=0.1   # system-bath coupling strength
nuc=5.0 # spectral density cut-off frequency
T=0.0   # bath temperature
jnu = lambda nu: 2*a*nu*np.exp(-(nu/nuc)**2) # Ohmic spectral density
bath=Bath(2,Sz,jnu,T) # Note couples via Sz

#-------- define control (unused in this example) ---
control=Control(dim=2)

# Start logging to mean-field.log
logging.info('Start TEMPO with mean-field: T={T}, gn {gn}, pp={pp}, K={K}, dt={dt} for '\
        '{end_step} steps.'.format(T=T, gn=gn, pp=pp, K=K, dt=dt, end_step=end_step))

#-------- RUN with tempo_sys ------------------------

start_time=time()
temposys=TempoSys(bath,system,control,initial_state,start_time=tS,dt=dt,dkmax=K,precision=pp,
        options={'name':'mean-field','backup_every_nth':0})
steps0,steps_runtime0=temposys.compute_dynamics(end_step)
# Get times and states from temposys
times,states=temposys.get_dynamics()
# Get spin expectations at each point in dynamics (or calculate from states)
t0,sz0=temposys.get_expectations(Sz,real=True)
# Get corresponding field values <a> from system.field_expectation_array
fields0=system.field_expectation_array
runtime0=round(time()-start_time,2)
print('tempo_sys: run completed in {}s.'.format(runtime0))

dt=0.05/3    # timestep
K=60*3        # memory cut-off
pp=80       # precision
tS=0.0      # start time
end_step=3*100# number of steps (tot. time = dt*end_step)
initial_state=Sx_up
w0=3.0
wc=3.0
system=System(dim=2, hamiltonian=Hsys, mean_field_info={'field_increment':field_increment,
    'initial_field':a0, 'interaction_operator': interaction_operator})

start_time=time()
temposys=TempoSys(bath,system,control,initial_state,start_time=tS,dt=dt,dkmax=K,precision=pp,
        options={'name':'mean-field','backup_every_nth':0})
steps0,steps_runtime0=temposys.compute_dynamics(end_step)
# Get times and states from temposys
times,states=temposys.get_dynamics()
# Get spin expectations at each point in dynamics (or calculate from states)
t1,sz1=temposys.get_expectations(Sz,real=True)
# Get corresponding field values <a> from system.field_expectation_array
fields1=system.field_expectation_array
runtime0=round(time()-start_time,2)
print('tempo_sys: run completed in {}s.'.format(runtime0))



#---------- Compare output ----------------------------

fig, axes = plt.subplots(2, sharex=True)
axes[1].set_xlabel('t')
axes[0].set_ylabel('n')
axes[1].set_ylabel('<Sz>')
axes[0].plot(t0*2.0, np.abs(fields0)**2, label='wc=2.0')
axes[0].plot(t1*3.0, np.abs(fields1)**2, label='wc=3.0')
axes[1].plot(t0*2.0, sz0, label='wc=2.0')
axes[1].plot(t1*3.0, sz1, label='wc=3.0')
axes[0].legend()
axes[1].legend()
plt.savefig('mean-field-process.png')

