<p align="center">
<img src="https://media.giphy.com/media/ftBzkYV06IWkGnfXRl/giphy.gif" width="5000" height="500" >
</p>

## Intorduction: using machine learning to learn kinetics of phase transitions

The phase transition is one of the most important concepts in materials science. In experiments, we are nowadays equipped with microcopies that have resolutions down to single atoms, and capable to record the process of phase transitions in in-situ experimental videos. The kinetics of phase transitions are hidden in these videos, and it is tedious and difficult to discover them manually. **How can use mahcine learning to assist us to learn kinetics of phase transitions from in-situ experimental data?** This is the question we want to explore in this project.  

## Phase field model
We have been using the phase-field model to model kinetics of phase transitions since several decades ago. The Allen-Cahn equation dictates that the temporal evolution of the phase field can be described by the partial-differential equation below,

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\phi}{\partial&space;t}&space;=&space;\kappa&space;\Delta&space;\phi&space;&plus;&space;\mu(\phi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\phi}{\partial&space;t}&space;=&space;\kappa&space;\Delta&space;\phi&space;&plus;&space;\mu(\phi)" title="\frac{\partial \phi}{\partial t} = \kappa \Delta \phi + \mu(\phi)" /></a>.

The left-hand side is the temporal derivative of the phase field. The first term on the right-hand side is the diffusive term containing the Laplace operator, and the second term is the chemical potential as a function of phase field. 

