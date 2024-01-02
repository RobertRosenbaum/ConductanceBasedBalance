import numpy as np

# E-I recurrent EIF spiking network with current-based external synaptic input
def EIFNetworkCurrentPlusFree(J, Sx, Jx, Ne, NeuronParameters, tau, Nt, dt, maxns, IeRecord, IiRecord):
  N = len(J)
  Ni = N - Ne

  Jee = J[:Ne, :Ne]
  Jei = J[:Ne, Ne:]
  Jie = J[Ne:, :Ne]
  Jii = J[Ne:, Ne:]
  Jex = Jx[:Ne, :]
  Jix = Jx[Ne:, :]

  Cm = NeuronParameters['Cm']
  gL = NeuronParameters['gL']
  EL = NeuronParameters['EL']
  Vth = NeuronParameters['Vth']
  Vre = NeuronParameters['Vre']
  Vlb = NeuronParameters['Vlb']
  DeltaT = NeuronParameters['DeltaT']
  VT = NeuronParameters['VT']

  taue = tau[0]
  taui = tau[1]
  taux = tau[2]

  Ve = np.random.rand(Ne) * (VT - Vre) + Vre
  Vi = np.random.rand(Ni) * (VT - Vre) + Vre

  VeFree = np.random.rand(Ne) * (VT - Vre) + Vre
  ViFree = np.random.rand(Ni) * (VT - Vre) + Vre

  Iee = np.zeros(Ne)
  Iei = np.zeros(Ne)
  Iie = np.zeros(Ni)
  Iii = np.zeros(Ni)
  Iex = np.zeros(Ne)
  Iix = np.zeros(Ni)

 # nBinsRecord = round(dtRecord / dt)
 # NtRec = int(np.ceil(Nt / nBinsRecord))

  Nerecord = len(IeRecord)
  VeRec = np.zeros((Nt, Nerecord))
  Nirecord = len(IiRecord)
  ViRec = np.zeros((Nt, Nirecord))

  VeFreeRec = np.zeros((Nt, Nerecord))
  ViFreeRec = np.zeros((Nt, Nirecord))

  IeeRec = np.zeros((Nt, Nerecord))
  IeiRec = np.zeros((Nt, Nerecord))
  IieRec = np.zeros((Nt, Nirecord))
  IiiRec = np.zeros((Nt, Nirecord))
  IexRec = np.zeros((Nt, Nerecord))
  IixRec = np.zeros((Nt, Nirecord))

  VeM=np.zeros(Ne)
  VeM2 = np.zeros(Ne)
  ViM=np.zeros(Ni)
  ViM2 = np.zeros(Ni)
  IeeM=np.zeros(Ne)
  IeeM2=np.zeros(Ne)
  IeiM=np.zeros(Ne)
  IeiM2=np.zeros(Ne)
  IieM=np.zeros(Ni)
  IieM2=np.zeros(Ni)
  IiiM=np.zeros(Ni)
  IiiM2=np.zeros(Ni)
  IexM=np.zeros(Ne)
  IexM2 = np.zeros(Ne)
  IixM=np.zeros(Ni)
  IixM2 = np.zeros(Ni)

  iXspike=0
  nspikeX=Sx.shape[1]

  nespike = 0
  nispike = 0
  TooManySpikes = False
  se = -1.0 + np.zeros((2, maxns))
  si = -1.0 + np.zeros((2, maxns))
  for i in range(Nt):
    # External inputs
    # Iex = Iex + dt * (-Iex + Jex @ Sx[:, i]) / taux
    # Iix = Iix + dt * (-Iix + Jix @ Sx[:, i]) / taux


    while iXspike + 1 < nspikeX and Sx[0,iXspike] <= i*dt:
      jpre = int(Sx[1,iXspike])
      Iex += Jex[:, jpre] / taux
      Iix += Jix[:, jpre] / taux
      iXspike += 1


    # Euler update to synaptic currents
    Iee -= dt * Iee / taue
    Iei -= dt * Iei / taui
    Iie -= dt * Iie / taue
    Iii -= dt * Iii / taui
    Iex -= dt * Iex / taux
    Iix -= dt * Iix / taux


    # Euler update to V
    Ve = Ve + (dt / Cm) * (Iee + Iei + Iex + gL * (EL - Ve) + DeltaT * np.exp((Ve - VT) / DeltaT))
    Vi = Vi + (dt / Cm) * (Iie + Iii + Iix + gL * (EL - Vi) + DeltaT * np.exp((Vi - VT) / DeltaT))
    Ve = np.maximum(Ve, Vlb)
    Vi = np.maximum(Vi, Vlb)

    VeFree = VeFree + (dt / Cm) * (Iee + Iei + Iex + gL * (EL - VeFree))
    ViFree = ViFree + (dt / Cm) * (Iie + Iii + Iix + gL * (EL - ViFree))
    # VeFree = np.maximum(VeFree, Vlb)
    # ViFree = np.maximum(ViFree, Vlb)


    # Find which E neurons spiked
    Ispike = np.nonzero(Ve >= Vth)[0]
    if Ispike.any() and not (TooManySpikes):
      # Store spike times and neuron indices
      if nespike + len(Ispike) <= maxns:
        se[0, nespike:nespike + len(Ispike)] = dt * i
        se[1, nespike:nespike + len(Ispike)] = Ispike
      else:
        TooManySpikes = True

      # Reset e mem pot.
      Ve[Ispike] = Vre

      # Update exc synaptic currents
      Iee = Iee + Jee[:, Ispike].sum(axis=1) / taue
      Iie = Iie + Jie[:, Ispike].sum(axis=1) / taue

      # Update cumulative number of e spikes
      nespike = nespike + len(Ispike)

    # Find which I neurons spiked
    Ispike = np.nonzero(Vi >= Vth)[0]
    if Ispike.any() and not (TooManySpikes):
      # Store spike times and neuron indices
      if nispike + len(Ispike) <= maxns:
        si[0, nispike:nispike + len(Ispike)] = dt * i
        si[1, nispike:nispike + len(Ispike)] = Ispike
      else:
        TooManySpikes = True

      # Reset i mem pot.
      Vi[Ispike] = Vre

      # Update inh synaptic currents
      Iei = Iei + Jei[:, Ispike].sum(axis=1) / taui
      Iii = Iii + Jii[:, Ispike].sum(axis=1) / taui

      # Update cumulative number of i spikes
      nispike = nispike + len(Ispike)

    if TooManySpikes:
      print('Too many spikes. Exiting sim at time t =', i * dt)
      break

    VeRec[i, :] = Ve[IeRecord]
    ViRec[i, :] = Vi[IiRecord]
    VeFreeRec[i, :] = VeFree[IeRecord]
    ViFreeRec[i, :] = ViFree[IiRecord]
    IeeRec[i,:] = Iee[IeRecord]
    IeiRec[i, :] = Iei[IeRecord]
    IieRec[i,:] = Iie[IiRecord]
    IiiRec[i, :] = Iii[IiRecord]
    IexRec[i, :] = Iex[IeRecord]
    IixRec[i, :] = Iix[IiRecord]

  Recording={}
  Recording['Ve']=VeRec
  Recording['Vi'] = ViRec
  Recording['VeFree'] = VeFreeRec
  Recording['ViFree'] = ViFreeRec

  Recording['Iee'] = IeeRec
  Recording['Iei'] = IeiRec
  Recording['Iie'] = IieRec
  Recording['Iii'] = IiiRec
  Recording['Iex'] = IexRec
  Recording['Iix'] = IixRec

  return se, si, Recording

# E-I recurrent EIF spiking network with conductance-based external synaptic input
def EIFNetworkCondPlusFree(J, Sx, Jx, Ne, NeuronParameters, tau, Nt, dt, maxns, IeRecord, IiRecord):
  N = len(J)
  Ni = N - Ne

  Cm = NeuronParameters['Cm']
  gL = NeuronParameters['gL']
  EL = NeuronParameters['EL']
  Vth = NeuronParameters['Vth']
  Vre = NeuronParameters['Vre']
  Vlb = NeuronParameters['Vlb']
  DeltaT = NeuronParameters['DeltaT']
  VT = NeuronParameters['VT']
  Ee = NeuronParameters['Ee']
  Ei = NeuronParameters['Ei']

  Vref = -65.0
  Jee = J[:Ne, :Ne] / (Ee - Vref)
  Jei = J[:Ne, Ne:] / (Ei - Vref)
  Jie = J[Ne:, :Ne] / (Ee - Vref)
  Jii = J[Ne:, Ne:] / (Ei - Vref)
  Jex = Jx[:Ne, :] / (Ee - Vref)
  Jix = Jx[Ne:, :] / (Ee - Vref)

  taue = tau[0]
  taui = tau[1]
  taux = tau[2]

  Ve = np.random.rand(Ne) * (VT - Vre) + Vre
  Vi = np.random.rand(Ni) * (VT - Vre) + Vre
  VeFree = np.random.rand(Ne) * (VT - Vre) + Vre
  ViFree = np.random.rand(Ni) * (VT - Vre) + Vre

  gee = np.zeros(Ne)
  gei = np.zeros(Ne)
  gie = np.zeros(Ni)
  gii = np.zeros(Ni)
  gex = np.zeros(Ne)
  gix = np.zeros(Ni)

  Nerecord = len(IeRecord)
  VeRec = np.zeros((Nt, Nerecord))
  Nirecord = len(IiRecord)
  ViRec = np.zeros((Nt, Nirecord))

  VeFreeRec = np.zeros((Nt, Nerecord))
  ViFreeRec = np.zeros((Nt, Nirecord))


  geeRec = np.zeros((Nt, Nerecord))
  geiRec = np.zeros((Nt, Nerecord))
  gieRec = np.zeros((Nt, Nirecord))
  giiRec = np.zeros((Nt, Nirecord))
  gexRec = np.zeros((Nt, Nerecord))
  gixRec = np.zeros((Nt, Nirecord))

  iXspike=0
  nspikeX=Sx.shape[1]

  nespike = 0
  nispike = 0
  TooManySpikes = False
  se = -1.0 + np.zeros((2, maxns))
  si = -1.0 + np.zeros((2, maxns))
  for i in range(Nt):
    # # External inputs
    # gex = gex + dt * (-gex + Jex @ Sx[:, i]) / taux
    # gix = gix + dt * (-gix + Jix @ Sx[:, i]) / taux

    while iXspike + 1 < nspikeX and Sx[0,iXspike] <= i*dt:
        jpre = int(Sx[1,iXspike])
        gex = gex + Jex[:, jpre] / taux
        gix = gix + Jix[:, jpre] / taux
        iXspike = iXspike + 1

    # Euler update to V
    Ve = Ve + (dt / Cm) * (gee * (Ee - Ve) + gei * (Ei - Ve) + gex * (Ee - Ve) + gL * (EL - Ve) + DeltaT * np.exp(
      (Ve - VT) / DeltaT))
    Vi = Vi + (dt / Cm) * (gie * (Ee - Vi) + gii * (Ei - Vi) + gix * (Ee - Vi) + gL * (EL - Vi) + DeltaT * np.exp(
      (Vi - VT) / DeltaT))
    Ve = np.maximum(Ve, Vlb)
    Vi = np.maximum(Vi, Vlb)

    VeFree = VeFree + (dt / Cm) * (gee * (Ee - VeFree) + gei * (Ei - VeFree) + gex * (Ee - VeFree) + gL * (EL - VeFree))
    ViFree = ViFree + (dt / Cm) * (gie * (Ee - ViFree) + gii * (Ei - ViFree) + gix * (Ee - ViFree) + gL * (EL - ViFree))
    #VeFree = np.maximum(VeFree, Vlb)
    #ViFree = np.maximum(ViFree, Vlb)

    # Find which E neurons spiked
    Ispike = np.nonzero(Ve >= Vth)[0]
    if Ispike.any() and not (TooManySpikes):
      # Store spike times and neuron indices
      if nespike + len(Ispike) <= maxns:
        se[0, nespike:nespike + len(Ispike)] = dt * i
        se[1, nespike:nespike + len(Ispike)] = Ispike
      else:
        TooManySpikes = True

      # Reset e mem pot.
      Ve[Ispike] = Vre

      # Update exc synaptic currents
      gee = gee + Jee[:, Ispike].sum(axis=1) / taue
      gie = gie + Jie[:, Ispike].sum(axis=1) / taue

      # Update cumulative number of e spikes
      nespike = nespike + len(Ispike)

    # Find which I neurons spiked
    Ispike = np.nonzero(Vi >= Vth)[0]
    if Ispike.any() and not (TooManySpikes):
      # Store spike times and neuron indices
      if nispike + len(Ispike) <= maxns:
        si[0, nispike:nispike + len(Ispike)] = dt * i
        si[1, nispike:nispike + len(Ispike)] = Ispike
      else:
        TooManySpikes = True

      # Reset i mem pot.
      Vi[Ispike] = Vre

      # Update inh synaptic currents
      gei = gei + Jei[:, Ispike].sum(axis=1) / taui
      gii = gii + Jii[:, Ispike].sum(axis=1) / taui

      # Update cumulative number of i spikes
      nispike = nispike + len(Ispike)

    if TooManySpikes:
      print('Too many spikes. Exiting sim at time t =', i * dt)
      break

    # Euler update to synaptic currents
    gee = gee - dt * gee / taue
    gei = gei - dt * gei / taui
    gie = gie - dt * gie / taue
    gii = gii - dt * gii / taui
    gex = gex - dt * gex / taux
    gix = gix - dt * gix / taux


    VeRec[i, :] = Ve[IeRecord]
    ViRec[i, :] = Vi[IiRecord]
    VeFreeRec[i, :] = VeFree[IeRecord]
    ViFreeRec[i, :] = ViFree[IiRecord]
    geeRec[i, :] = gee[IeRecord]
    geiRec[i, :] = gei[IeRecord]
    gieRec[i, :] = gie[IiRecord]
    giiRec[i, :] = gii[IiRecord]
    gexRec[i, :] = gex[IeRecord]
    gixRec[i, :] = gix[IiRecord]

  Recording = {}
  Recording['Ve'] = VeRec
  Recording['Vi'] = ViRec
  Recording['VeFree'] = VeFreeRec
  Recording['ViFree'] = ViFreeRec

  Recording['gee'] = geeRec
  Recording['gei'] = geiRec
  Recording['gie'] = gieRec
  Recording['gii'] = giiRec
  Recording['gex'] = gexRec
  Recording['gix'] = gixRec

  return se, si, Recording



# Function to generate blockwise ER connection matrix
# NsPre = tuple of ints containing number of pre neurons in each block
# Jm = matrix connection weights in each block
# P = matrix of connection probs in each block
# NsPost = number of post neurons in each block
# If NsPost == None, connectivity is assumed recurrent (so NsPre=NsPost)
def GetBlockErdosRenyi(NsPre,Jm,P,NsPost=None):

  if NsPost==None:
    NsPost=NsPre

  # # If Jm is a 1D array, reshape it to column vector
  # if len(Jm.shape)==1:
  #   Jm = np.array([Jm]).T
  # if len(P.shape)==1:
  #   P = np.array([P]).T

  Npre = int(np.sum(NsPre))
  Npost = int(np.sum(NsPost))
  cNsPre = np.cumsum(np.insert(NsPre,0,0)).astype(int)
  cNsPost = np.cumsum(np.insert(NsPost,0,0)).astype(int)
  J = np.zeros((Npost,Npre))

  for j1,N1 in enumerate(NsPost):
    for j2,N2 in enumerate(NsPre):
      J[cNsPost[j1]:cNsPost[j1+1],cNsPre[j2]:cNsPre[j2+1]]=Jm[j1,j2]*(np.random.binomial(1, P[j1,j2], size=(N1, N2)))
  return J


def PoissonProcess(r,Nt,dt,n=1,c=0,rep='full',taujitter=0):
  T=Nt*dt

  if c == 0 or n == 1:
    print('r', c, n)
    S = np.random.binomial(1, r * dt, (n, Nt)) / dt
  else:
    print('m')
    rm = r / c
    if rm * dt > .05:
      print('warning: mother process rate is kinda large')
    Sm = np.random.binomial(1, c, (n, Nt)) / dt
    S = Sm * np.random.binomial(1, rm * dt, Nt)
  if rep == 'sparse':
    I, J = np.nonzero(S)
    SpikeTimes = J * dt
    NeuronInds = I

    if taujitter>0:
      SpikeTimes = SpikeTimes + taujitter * np.random.randn(len(SpikeTimes))
      SpikeTimes[SpikeTimes<0] = -SpikeTimes[SpikeTimes<0]
      SpikeTimes[SpikeTimes>T] = T - (SpikeTimes[SpikeTimes>T] - T)

    Isort = np.argsort(SpikeTimes)
    SpikeTimes = SpikeTimes[Isort]
    NeuronInds = NeuronInds[Isort]
    ns = len(SpikeTimes)
    S = np.zeros((2, ns))
    S[0, :] = SpikeTimes
    S[1, :] = NeuronInds

  if taujitter!=0 and (c==0 or rep=='full'):
    print('Cannot jitter')


  # elif rep=='sparse':
  #   if c==0 or n==1:
  #     T=Nt*dt
  #     ns=np.random.poisson(r*T)
  #     SpikeTimes=np.sort(np.random.rand(ns)*T)
  #     NeuronInds=np.random.randint(n)
  #     # Isort=np.argsort(SpikeTimes)
  #     # SpikeTimes=SpikeTimes[Isort]
  #     # NeuronInds=NeuronInds[Isort]
  #     S=np.zeros((2,ns))
  #     S[0,:]=SpikeTimes
  #     S[1,:]=NeuronInds
  #   else:
  #     # First generate full, then convert to sparse
  #     # Maybe fix this later to generate sparse from the start.
  #     rm = r / c
  #     if rm*dt > .05:
  #         print('warning: mother process rate is kinda large')
  #     Sm = np.random.binomial(1, rm * dt, (n, Nt))/dt
  #     S = Sm * np.random.binomial(1, c, (n, Nt))
  #     I, J = np.nonzero(S)
  #     SpikeTimes = J * dt
  #     NeuronInds = I
  #     Isort=np.argsort(SpikeTimes)
  #     SpikeTimes=SpikeTimes[Isort]
  #     NeuronInds=NeuronInds[Isort]
  #     ns=len(SpikeTimes)
  #     S=np.zeros((2,ns))
  #     S[0,:]=SpikeTimes
  #     S[1,:]=NeuronInds

  return S



# # Returns 2D array of spike counts from sparse spike train, s.
# # Counts spikes over window size winsize.
# # h is represented as (neuron)x(time)
# # so h[j,k] is the spike count of neuron j at time window k
# def GetSpikeCounts(s,winsize,N,T):
#
#   xedges=np.arange(0,N+1,1)
#   yedges=np.arange(0,T+winsize,winsize)
#   h,_,_=np.histogram2d(s[1,:],s[0,:],bins=[xedges,yedges])
#   return h
#
# # Returns a resampled version of x
# # with a different dt.
# def DumbDownsample(x,dt_old,dt_new):
#   n = int(dt_new/dt_old)
#   if n<=1:
#     print('New dt should be larger than old dt. Returning x.')
#     return x
#   return x.reshape()
