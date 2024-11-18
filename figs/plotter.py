import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.figure('AFOIL')

x_exact, Cp_exact = tuple(np.loadtxt(r'Afoil_surface_data.dat',skiprows=1).T)[0:2]
plt.plot(x_exact,Cp_exact,'k--',label='exact')

# data = np.load(r'AFOIL_raw_0o1%c.npz')
# plt.plot(data['x'],data['Cp'],label='PIV raw data')

data = np.load(r'AFOIL_extrap_2%c_2%noise.npz')
plt.plot(data['x'],data['Cp'],label='PIV + linear extrapolation')

data = np.load(r'AFOIL_streamlines_3%c_2%noise.npz')
plt.plot(data['x'],data['Cp'],label='PIV + streamline method')

plt.xlabel('x/c'); plt.ylabel('Cp'); plt.legend(); plt.grid(); plt.gca().invert_yaxis(); plt.title('A-Airfoil')

plt.savefig('Afoil.png',dpi=500)





plt.figure('CYLINDER')

data = pd.read_excel(r'10_8_24.xlsx')
theta_taps = data['theta.1'].to_numpy()
Cp_taps = data['Cp.1'].to_numpy()
plt.errorbar(theta_taps, Cp_taps, yerr=0.03, fmt='k.', mfc='none', barsabove=True, label='taps', capsize=5)


# data = np.load(r'CYLINDER_raw_0mm.npz')
# plt.plot(data['theta'],data['Cp'],label='PIV raw data')

data = np.load(r'CYLINDER_extrap_2mm.npz')
plt.plot(data['theta'],data['Cp'],label='PIV + linear extrapolation')

data = np.load(r'CYLINDER_streamlines_2mm.npz')
plt.plot(data['theta'],data['Cp'],label='PIV + streamline method')

plt.xlabel(r'$\theta\;[^\circ]$'); plt.ylabel('Cp'); plt.legend(); plt.grid(); plt.title('Cylinder')

plt.savefig('cylinder.png',dpi=500)





plt.show()