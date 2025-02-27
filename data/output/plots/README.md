# Output figures

## Figure 1
We plot the semi-analytical solutions for Gubser flow with conserved charges in the plane $y=0$ (except $\pi^{xy}$, which is plotted in the plane $y=x$).
Included are (a) the temperature, (b) chemical potential, (c) the dimensionless quantity $\pi^{yy}/w$, where $w=\mathcal E + \mathcal P$ is the enthalpy density, for a diagonal entry of the shear stress tensor, and (d) the dimensionless quantity $\pi^{xy}/w$ for an off-diagonal entry.
The three different colors correspond to different initial conditions for the ratio of initial temperature and initial chemical potential: $\hat{\mu}_{Y,0}/\hat{T}_0=0$ (black), 1.5 (red), 3 (blue).
The different types of lines correspond to selected times $\tau=1.2$ fm/$c$ (solid lines), $2.0$ fm/$c$ (dashed lines) and $3.0$ fm/$c$ (dotted lines) during the evolution.

## Figure 2
We plot (a) the energy density, (b) number density, and (c) entropy density corresponding to the semi-analytical solutions for viscous Gubser flow with conserved charges shown in Figure 1 in the plane $y=0$.
We include the results for differential values of the initial conditions and at different times.
The number density plot does not feature black lines because at $\hat{\mu}_{Y,0}/\hat{T}_0=0$ the system is at exactly vanishing number densities such that there is no evolution of the number density.
Most notably, an increase in chemical potential can significantly decrease spatial variations of the energy
density and entropy.
The entropy density is most sensitive to the details of the temperature profile, as can be 
seen by the presence of the shoulders for the blue lines $(\hat{\mu}_{Y,0}/\hat{T}_0 = 3)$.

## Figure 3
We plot (a) the freeze-out surface and normal vectors, and $T$ versus $\mu_Y$ trajectories for two initial conditions, (b) $\hat \mu_{Y,0}/ \hat T_0 = 2$ and (c) $\hat \mu_{Y,0}/ \hat T_0 = 3$ with $\hat T_0 = 1.2$.
Included in the freeze-out plot is a colorbar which indicates the entropy density for a freeze-out cell $(\tau_\mathrm{FO}, r)$.
As expected, at zero chemical potential, the freeze-out surface is isentropic.
At non-vanishing chemical potential, the entropy in the core is lower (i.e., lower temperature) than the entropy at the tails.
The absolute maximum magnitude of the normal vectors also grows with increasing chemical potential, indicating that the particles emitted from this freeze-out surface can be more boosted.
To allow the normal vectors and freeze-out surface on the screen, they have been rescaled by a factor of $0.05$.

## Figure 4
We plot the numerical solution from the CCAKE code (dots) and semi-analytical solution (solid line) 
using EoS2 as a function of distance from the origin and for various time steps (indicated by the colorbar) within the first fm/$c$ of evolution.
The comparison is made between (a) energy density, (b) number density, (c) the $x$-component of the fluid velocity, (d) the $xx$-component of the shear stress tensor, (e) 
the inverse Reynolds number, and (f) the $xy$-component of the shear stress tensor.
We see excellent agreement for all shown variables with a very mild deviation of around a percent around the core for the energy density and number density.
For the $xx$-component of the shear stress tensor, we also observe an increasing deviation of the numerical solution from the semi-analytical one for increasing time, specifically between $r=1$ fm and $r=2$ fm, with the maximum relative deviations at $\tau = 1.6$ fm/$c$ getting large since the numerical solution is very close to zero. The largest deviations are seen on the $xy$-component of the shear stress tensor between $r=1$ fm and $r=2$ fm, in similar fashion to the diagonal component.

## Figure 5
A comparison between the freeze-out surface and normal vectors for CCAKE (red) and VGCC (black). We see a slight disagreement between both the freeze-out time and the direction of the freeze-out vector.
This disagreement is mostly like a result of the SPH particles freezing out a time-step after the actual freeze-out time within the simulation.
