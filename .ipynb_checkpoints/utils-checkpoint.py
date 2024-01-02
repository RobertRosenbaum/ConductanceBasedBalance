import numpy as np

# E-I recurrent EIF spiking network with current-based external synaptic input
def EIFNetworkCurrent(J, Sx, Jx, Ne, NeuronParameters, tau, Nt, dt, maxns, IeRecord, IiRecord):
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
    Iee = np.zeros(Ne)
    Iei = np.zeros(Ne)
    Iie = np.zeros(Ni)
    Iii = np.zeros(Ni)
    Iex = np.zeros(Ne)
    Iix = np.zeros(Ni)

    Nerecord = len(IeRecord)
    VeRec = np.zeros((Nt, Nerecord))
    Nirecord = len(IiRecord)
    ViRec = np.zeros((Nt, Nirecord))

    nespike = 0
    nispike = 0
    TooManySpikes = False
    se = -1.0 + np.zeros((2, maxns))
    si = -1.0 + np.zeros((2, maxns))
    for i in range(Nt):
        # External inputs
        Iex = Iex + dt * (-Iex + Jex @ Sx[:, i]) / taux
        Iix = Iix + dt * (-Iix + Jix @ Sx[:, i]) / taux

        # Euler update to V
        Ve = Ve + (dt / Cm) * (Iee + Iei + Iex + gL * (EL - Ve) + DeltaT * np.exp((Ve - VT) / DeltaT))
        Vi = Vi + (dt / Cm) * (Iie + Iii + Iix + gL * (EL - Vi) + DeltaT * np.exp((Vi - VT) / DeltaT))
        Ve = np.maximum(Ve, Vlb)
        Vi = np.maximum(Vi, Vlb)

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

        # Euler update to synaptic currents
        Iee = Iee - dt * Iee / taue
        Iei = Iei - dt * Iei / taui
        Iie = Iie - dt * Iie / taue
        Iii = Iii - dt * Iii / taui

        VeRec[i, :] = Ve[IeRecord]
        ViRec[i, :] = Vi[IiRecord]

    print(Iex.mean())

    return se, si, VeRec, ViRec


# E-I recurrent EIF spiking network with conductance-based external synaptic input
def EIFNetworkCond(J, Sx, Jx, Ne, NeuronParameters, tau, Nt, dt, maxns, IeRecord, IiRecord):
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

    Vref = EL
    Jee = J[:Ne, :Ne]/(Ee-Vref)
    Jei = J[:Ne, Ne:]/(Ei-Vref)
    Jie = J[Ne:, :Ne]/(Ee-Vref)
    Jii = J[Ne:, Ne:]/(Ei-Vref)
    Jex = Jx[:Ne, :]/(Ee-Vref)
    Jix = Jx[Ne:, :]/(Ee-Vref)



    taue = tau[0]
    taui = tau[1]
    taux = tau[2]

    Ve = np.random.rand(Ne) * (VT - Vre) + Vre
    Vi = np.random.rand(Ni) * (VT - Vre) + Vre
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

    nespike = 0
    nispike = 0
    TooManySpikes = False
    se = -1.0 + np.zeros((2, maxns))
    si = -1.0 + np.zeros((2, maxns))
    for i in range(Nt):
        # External inputs
        gex = gex + dt * (-gex + Jex @ Sx[:, i]) / taux
        gix = gix + dt * (-gix + Jix @ Sx[:, i]) / taux

        # Euler update to V
        Ve = Ve + (dt / Cm) * (gee*(Ee-Ve) + gei*(Ei-Ve) + gex*(Ee-Ve) + gL*(EL - Ve) + DeltaT * np.exp((Ve - VT) / DeltaT))
        Vi = Vi + (dt / Cm) * (gie*(Ee-Vi) + gii*(Ei-Vi) + gix*(Ee-Vi) + gL * (EL - Vi) + DeltaT * np.exp((Vi - VT) / DeltaT))
        Ve = np.maximum(Ve, Vlb)
        Vi = np.maximum(Vi, Vlb)

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

        VeRec[i, :] = Ve[IeRecord]
        ViRec[i, :] = Vi[IiRecord]


    return se, si, VeRec, ViRec
