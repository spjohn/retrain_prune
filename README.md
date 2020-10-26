This repository contains the codes that were used to get the simulation results present in the journal paper titled: "**Retraining a Pruned Network: A Unified Theory of Time Complexity**" which was published in Springer journal "SN Computer Science" on 18 June 2020.

**Authors: Soumya Sara John, Dr. Deepak Mishra, Dr. Sheeba Rani J.**

The journal paper can be found [here](https://link.springer.com/article/10.1007/s42979-020-00208-w)

#### Abstract:
Fine-tuning of neural network parameters is an essential step that is involved in model compression via pruning, which let the network relearn using the training data. The time needed to relearn a compressed neural network model is crucial in identifying a hardware-friendly architecture. This paper analyzes the fine-tuning or retraining step after pruning the network layer-wise and derives lower and upper bounds for the number of iterations; the network will take for retraining the pruned network till the required error of convergence. The bounds are defined with respect to the desired convergence error, optimizer parameters, amount of pruning and the number of iterations for the initial training the network. Experiments on LeNet-300-100 and LeNet-5 networks validate the bounds defined in the paper for pruning using the random connection pruning and clustered pruning approaches.
