"""GEMPAK calculations."""

import numpy as np

# Gravitational constant
g = 9.80616  # m / s^2

# Dry air gas constant
Rd = 287.04  # J / K / kg


def mixing_ratio(dwpc, pres, missing=-9999):
    """Calculate the water vapor mixing ratio.

    Parameters
    ----------
    dwpc : float
        Dewpoint (degC)

    pres : float
        Total air pressure (hPa)

    missing : float, optional
        Missing data flag

    Returns
    -------
    float
        The (mass) mixing ratio (kg/kg)

    Notes
    -----
    See GEMPAK function PR_MIXR
    """
    if dwpc == missing or pres == missing:
        mixr = missing
    else:
        vapr = vapor_pressure(dwpc, missing)
        if vapr == missing:
            mixr = missing
        else:
            corr = (1.001 + ((pres - 100.) / 900.) * 0.0034)
            e = corr * vapr
            if e > (0.5 * pres):
                mixr = missing
            else:
                mixr = 0.62197 * (e / (pres - e)) * 1000.
    return mixr


def moist_hydrostatic_height(z_bot, pres_bot, pres_top, scale_height,
                             missing=-9999):
    """Calculate the moist hydrostatic height at the top of a layer.

    Parameters
    ----------
    z_bot : float
        Bottom of layer height (m)

    pres_bot : float
        Bottom of layer pressure (hPa)

    pres_top : float
        Top of layer pressure (hPa)

    sacle_height : float
        Scale height of layer (m)

    missing : float, optional
        Missing data flag

    Returns
    -------
    float
        The moist hydrostatic height (m)

    Notes
    -----
    See GEMPAK function PR_MHGT
    """
    if (z_bot == missing or pres_bot == missing
       or pres_top or scale_height == missing):
        mhgt = missing
    else:
        mhgt = z_bot + scale_height * np.loa(pres_bot / pres_top)
    return mhgt


def scale_height(tmpc_bot, tmpc_top, dwpc_bot, dwpc_top,
                 pres_bot, pres_top, missing=-9999):
    """Calculate the scale height of a layer.

    Parameters
    ----------
    tmpc_bot : float
        Bottom of layer temperature (degC)

    tmpc_top : float
        Top of layer temperature (degC)

    dwpc_bot : float
        Bottom of layer dewpoint (degC)

    dwpc_top : float
        Top of layer dewpoint (degC)

    pres_bot : float
        Bottom of layer pressure (hPa)

    pres_top : float
        Top of layer pressure (hPa)

    missing : float, optional
        Missing data flag

    Returns
    -------
    float
        The (mass) mixing ratio (kg/kg)

    Notes
    -----
    See GEMPAK function PR_SCLH
    """
    if (tmpc_bot == missing
       or tmpc_top == missing
       or dwpc_bot == missing
       or dwpc_top == missing
       or pres_bot == missing
       or pres_top == missing):
        sclh = missing
    else:
        tvbk = virtual_temperature(tmpc_bot, pres_bot, missing)
        tvtk = virtual_temperature(tmpc_top, pres_top, missing)
        if tvbk == missing or tvtk == missing:
            sclh = missing
        else:
            tavg = (tvbk + tvtk) * 0.5
            sclh = (Rd / g) * tavg
    return sclh


def vapor_pressure(dwpc, missing=-9999):
    """Calculate the partial water vapor pressure.

    Parameters
    ----------
    dwpc : float
        Dewpoint (degC)

    missing : float, optional
        Missing data flag

    Returns
    -------
    float
        The partial pressure of water vapor (hPa)

    Notes
    -----
    See GEMPAK function PR_VAPR
    """
    if dwpc == missing:
        vapr = missing
    else:
        vapr = 6.112 * np.exp((17.67 * dwpc) / (dwpc + 243.5))
    return vapr


def virtual_temperature(tmpc, dwpc, pres, missing=-9999):
    """Calculate the virtual temperature.

    Parameters
    ----------
    tmpc : float
        Temperature (degC)

    dwpc : float
        Dewpoint (degC)

    pres : float
        Air pressure (hPa)

    missing : float, optional
        Missing data flag

    Returns
    -------
    float
        The virtual temperature (K)

    Notes
    -----
    See GEMPAK function PR_TVRK
    """
    if tmpc == missing or pres == missing:
        tvirt = missing
    elif dwpc == missing:
        tvirt = tmpc + 273.15
    else:
        tmpk = tmpc + 273.15
        rmix = mixing_ratio(dwpc, pres, missing)
        if rmix == missing:
            tvirt = tmpc + 273.15
        else:
            tvirt = tmpk * (1. + 0.001 * rmix / 0.62197) / (1. + 0.001 * rmix)
    return tvirt


def interp_logp_data(sounding, missing=-9999):
    """Interpolate missing data with respect to log p.

    This function is similar to the MR_MISS subroutine
    in GEMPAK and also incorporates PC_INTP functionality.
    """
    size = len(sounding['PRES'])
    recipe = [('TEMP', 'DWPT'), ('DRCT', 'SPED'), ('DWPT', None)]

    for var1, var2 in recipe:
        iabove = 0
        i = 1
        more = True
        while (i < size - 1) and more:
            if sounding[var1][i] == missing:
                if iabove <= i:
                    iabove = i + 1
                    found = False
                    while not found:
                        if sounding[var1][iabove] != missing:
                            found = True
                        else:
                            iabove += 1
                            if iabove >= size:
                                found = True
                                iabove = 0
                                more = False

                if var2 is None and iabove != 0:
                    if sounding['PRES'][i] - sounding['PRES'][iabove] > 100:
                        for j in range(iabove, size):
                            sounding['DWPT'][j] = missing
                        iabove = 0
                        more = False

                if (var1 == 'DRCT' and iabove != 0
                   and sounding['PRES'][i - 1] > 100
                   and sounding['PRES'][iabove] < 100):
                    iabove = 0

                if iabove != 0:
                    output = {}
                    pres1 = sounding['PRES'][i - 1]
                    pres2 = sounding['PRES'][iabove]
                    vlev = sounding['PRES'][i]
                    between = (((pres1 < pres2) and (pres1 < vlev)
                               and (vlev < pres2))
                               or ((pres2 < pres1) and (pres2 < vlev)
                               and (vlev < pres1)))
                    if not between:
                        raise RuntimeError('Cannot interpolate sounding data.')
                    elif pres1 <= 0 or pres2 <= 0:
                        raise ValueError('Pressure cannot be negative.')

                    rmult = np.log(vlev / pres1) / np.log(pres2 / pres1)
                    output['PRES'] = vlev
                    for parm in [var1, var2]:
                        if parm is None:
                            continue
                        output[parm] = missing
                        if parm == 'DRCT':
                            angle1 = sounding[parm][i - 1] % 360
                            angle2 = sounding[parm][iabove] % 360
                            if abs(angle1 - angle2) > 180:
                                if angle1 < angle2:
                                    angle1 -= 360
                                else:
                                    angle2 -= 360
                            iangle = angle1 + (angle2 - angle1) * rmult
                            output[parm] = iangle % 360
                        else:
                            output[parm] = (
                                sounding[parm][i - 1]
                                + (sounding[parm][iabove] - sounding[parm][i - 1]) * rmult
                            )
                    sounding[var1][i] = output[var1]
                    if var2 is not None:
                        sounding[var2][i] = output[var2]
            i += 1


def interp_logp_height(sounding, missing=-9999):
    """Interpolate height linearly with respect to log p.

    This function mimics the functionality of the MR_INTZ
    subroutine in GEMPAK.
    """
    size = len(sounding['HGHT'])

    idx = -1
    maxlev = -1
    while size + idx != 0:
        if sounding['HGHT'][idx] != missing:
            maxlev = size + idx
            break
        else:
            idx -= 1

    pbot = missing
    for i in range(maxlev):
        pres = sounding['PRES'][i]
        hght = sounding['HGHT'][i]

        if pres == missing:
            continue
        elif hght != missing:
            pbot = pres
            zbot = hght
            ptop = 2000
        elif pbot == missing:
            continue
        else:
            ilev = i + 1
            while pres <= ptop:
                if sounding['HGHT'][ilev] != missing:
                    ptop = sounding['PRES'][ilev]
                    ztop = sounding['HGHT'][ilev]
                else:
                    ilev += 1
            sounding['HGHT'][i] = (zbot + (ztop - zbot)
                                   * (np.log(pres / pbot) / np.log(ptop / pbot)))

    if maxlev < size - 1:
        if maxlev > -1:
            pb = sounding['PRES'][maxlev]
            zb = sounding['HGHT'][maxlev]
            tb = sounding['TEMP'][maxlev]
            tdb = sounding['DWPT'][maxlev]
        else:
            pb = missing
            zb = missing
            tb = missing
            tdb = missing

        for i in range(maxlev + 1, size):
            if sounding['HGHT'][i] == missing:
                tt = sounding['TEMP'][i]
                tdt = sounding['DWPT'][i]
                pt = sounding['PRES'][i]
                H = scale_height(tb, tt, tdb, tdt, pb, pt, missing)
                sounding['HGHT'][i] = moist_hydrostatic_height(zb, pb, pt, H)


def interp_logp_pressure(sounding, missing=-9999):
    """Interpolate pressure from heights.

    This function is similar to the MR_INTP subroutine from GEMPAK.
    """
    i = 0
    ilev = -1
    klev = -1
    size = len(sounding['PRES'])
    pb = missing
    zb = missing

    while i < size:
        p = sounding['PRES'][i]
        z = sounding['HGHT'][i]

        if p != missing and z != missing:
            klev = i
            pt = p
            zt = z

        if ilev != -1 and klev != -1:
            for j in range(ilev + 1, klev):
                z = sounding['HGHT'][j]
                if z != missing:
                    sounding['PRES'][j] = (
                        pb * np.exp((z - zb) * np.log(pt / pb) / (zt - zb))
                    )
        ilev = klev
        pb = pt
        zb = zt
        i += 1


def interp_moist_height(sounding, missing=-9999):
    """Interpolate moist hydrostatic height.

    This function mimics the functionality of the MR_SCMZ
    subroutine in GEMPAK. This the default behavior when
    merging observed sounding data.
    """
    hlist = (np.ones(len(sounding['PRES'])) * -9999)

    ilev = -1
    top = False

    found = False
    while not found and not top:
        ilev += 1
        if ilev >= len(sounding['PRES']):
            top = True
        elif (sounding['PRES'][ilev] != missing
              and sounding['TEMP'][ilev] != missing
              and sounding['HGHT'][ilev] != missing):
            found = True

    while not top:
        pb = sounding['PRES'][ilev]
        plev = sounding['PRES'][ilev]
        tb = sounding['TEMP'][ilev]
        tdb = sounding['DWPT'][ilev]
        zb = sounding['HGHT'][ilev]
        zlev = sounding['HGHT'][ilev]
        jlev = ilev
        klev = 0
        mand = False

        while not mand:
            jlev += 1
            if jlev >= len(sounding['PRES']):
                mand = True
                top = True
            else:
                pt = sounding['PRES'][jlev]
                tt = sounding['TEMP'][jlev]
                tdt = sounding['DWPT'][jlev]
                zt = sounding['HGHT'][jlev]
                if (zt != missing
                   and tt != missing):
                    mand = True
                    klev = jlev
                if (sounding['PRES'][ilev] != missing
                   and sounding['TEMP'][ilev] != missing
                   and sounding['PRES'][jlev] != missing
                   and sounding['TEMP'][jlev] != missing):
                    H = scale_height(tb, tt, tdb, tdt, pb, pt, missing)
                    znew = moist_hydrostatic_height(zb, pb, pt, H)
                    tb = tt
                    tdb = tdt
                    pb = pt
                    zb = znew
                else:
                    H = missing
                    znew = missing
                hlist[jlev] = H

        if klev != 0:
            s = (zt - zlev) / (znew - zlev)
            for h in range(ilev + 1, klev + 1):
                hlist[h] *= s

        hbb = zlev
        pbb = plev
        for ii in range(ilev + 1, jlev):
            p = sounding['PRES'][ii]
            H = hlist[ii]
            z = moist_hydrostatic_height(hbb, pbb, p, H)
            sounding['HGHT'][ii] = z
            hbb = z
            pbb = p

        ilev = klev
