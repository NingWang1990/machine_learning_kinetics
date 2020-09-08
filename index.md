<p align="center">
<img src="https://media.giphy.com/media/ftBzkYV06IWkGnfXRl/giphy.gif" width="5000" height="500" >
</p>

## Intorduction: using deep learning to learn kinetics of phase transitions

The phase transition is one of the most important concept in materials science. In experiments, we are nowadays capable to observe the process of phase transitions in transmission electron microscopy that has resolution down to single atoms.  In the theoretical aspect, we have been using the phase-field model to model kinetics of phase transitions for a long time. 



The phase-field equation (Allen-Cahn) dictates that the temporal evolution of the phase field can be described by the partial-differential equation below,

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\phi}{\partial&space;t}&space;=&space;\kappa&space;\Delta&space;\phi&space;&plus;&space;\mu(\phi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\phi}{\partial&space;t}&space;=&space;\kappa&space;\Delta&space;\phi&space;&plus;&space;\mu(\phi)" title="\frac{\partial \phi}{\partial t} = \kappa \Delta \phi + \mu(\phi)" /></a>.

The left-hand side is the temporal derivative of the phase field. The first term on the right-hand side is the diffusive term containing Laplace operator, and the second term is the chemical potential as a function of phase field. 

