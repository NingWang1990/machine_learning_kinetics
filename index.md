
# Using machine learning to learn kinetics of phase transitions


<p align="center">
<img src="https://media.giphy.com/media/ftBzkYV06IWkGnfXRl/giphy.gif" width="5000" height="500" >
</p>

visualize results with better resolution [here](https://drive.google.com/file/d/1QyNy-73R8cHTt2BE9qUn5ptpvA2tau5Q/view). 

## Intorduction

The phase transition is one of the most important concepts in materials science. In experiments, we are nowadays equipped with microcopies that have resolutions down to single atoms, and capable to record the process of phase transitions in in-situ experimental videos. The kinetics of phase transitions are hidden in these videos, and it is tedious and difficult to discover them manually. **How can use mahcine learning to assist us to learn kinetics of phase transitions from in-situ experimental data?** This is the question we want to explore in this project. To be more specific, we try to learn a phase-field model from the in-situ experimental video.

## Phase-field model
We have been using the phase-field method to model kinetics of phase transitions since several decades ago. The Allen-Cahn equation dictates that the temporal evolution of the phase field can be described by the partial-differential equation below,

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\phi}{\partial&space;t}&space;=&space;\kappa&space;\Delta&space;\phi&space;&plus;&space;\mu(\phi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\phi}{\partial&space;t}&space;=&space;\kappa&space;\Delta&space;\phi&space;&plus;&space;\mu(\phi)" title="\frac{\partial \phi}{\partial t} = \kappa \Delta \phi + \mu(\phi)" /></a>.

The left-hand side is the temporal derivative of the phase field. The first term on the right-hand side is the diffusive term containing the Laplace operator, and the second term is the chemical potential as a function of phase field. 
To learn a phase-field model means that to **learn the coefficient and the chemical potential from the experimental video**.


## Physics-informed neural network
We employ the physics-informed neural network (Raissi et al., J. Comput. Phys. 2019) in this project. As the name dictates, the neural network is informed with the physics. In this work, the physics is the phase-field model that describes kinetics of phase transitions. A schematic diagram of thie neural network is shown in the plot at the top of this webpage.

## Results
The main result of this work is shown in the plot at the top of this webpage. The result with better resolution can be visualized [here](https://drive.google.com/file/d/1QyNy-73R8cHTt2BE9qUn5ptpvA2tau5Q/view). 
We employed **pytorch** from facebook to construct our neural network, and train it on NVIDIA Tesla Volta GPU 32GB. Our training data are the experimental in-situ video of the phase seperation in a high-entropy alloy, the left one in the plot. One output of the neural network is the denoised video, the middle one in the plot. The other output is the learnt phase-field model. We showed the learning process of the chemical potential in the right video in the plot. 

