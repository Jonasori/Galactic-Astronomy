


from astropy.coordinates import SkyCoord


# Cluster:
# vector = [name, feh, ra, dec, pos_sig, pm_ra, pm_dec, pm_sig, parallax, parallax_sig]



# Omega Centauri: Globular Cluster
# Fe/H = -1.7 (https://arxiv.org/pdf/astro-ph/0605612.pdf); -1.5 is as close as we get here.
isochrones_omegaCen='./isochrones/fehm15afep0.Gaia'
coords_OmegaCen = SkyCoord('13h26m47.28s', '47d28m46.1s', frame='icrs', unit='deg')
ra_OmegaCen, dec_OmegaCen = [float(i) for i in coords_OmegaCen.to_string().split()]
d_OmegaCen = 4844.302     # 15.8kly in PC
vector_OmegaCen = ['Omega Centauri', isochrones_omegaCen,    # Name
                   ra_OmegaCen, dec_OmegaCen, 4,             # RA, DEC, radius
                   1.6, -0.15, 3,                              # pm (RA), pm (dec), radius
                   1/d_OmegaCen, 2]                          # Parallax, radius



# M43: Open Cluster (Orion Nebula)
isochrones_m43='./isochrones/fehp00afep0.Gaia'
coords_m43 = SkyCoord('05h35m31.8s', '−05d17m57s', frame='icrs', unit='deg')
ra_m43, dec_m43 = [float(i) for i in coords_m43.to_string().split()]
d_m43 = 490.5622     # 1600ly in pc
vector_m43 = ['M43', isochrones_m43,
              ra_m43, dec_m43, 2,
              3, 0, 8,
              1/d_m43, 2]
