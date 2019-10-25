# 2019_WNTR_Surrogate_Model

In the “Code” folder, I set up a script “MAIN.py”, which is needed to run the simulations and can be the starting point for everything else. At this stage, the script allows to: 
Load and represent the c-town network (or any other network among those available, but right now I set it up for the c-town network
change the demand patterns at the network junctions (with some randomness). This is done via the methods “randomlyScaleMultipliers” and “randomlyShiftMultipliers”, which I added to the “testWN.py” file (is the one that Luo prepared, but I defined the two functions above and I think they are the only one used right now in the main file. The first function allows modifying the amplitude of the demand multipliers of a function by a maximum percentage defined by an input value now set to +/-10%, with a probability of 20%. The second one shifts the multipliers (with probability 20%) by an input time window that is now set to +/- 3 hours.
Run the simulation. The simulation is now running with a start-and-stop mode, which means it is in a loop and each loop iterations advances by one simulation step (1 hour, in our case). For more details, see here (https://wntr.readthedocs.io/en/latest/hydraulics.html#pause-and-restart). The simulation also includes a simplified version of water quality, with the water age option (https://wntr.readthedocs.io/en/latest/waterquality.html). According to the EPANET guidelines "Water age provides a simple, non-specific measure of the overall quality of delivered drinking water” (https://www.microimages.com/documentation/Tutorials/Epanet2UserManual.pdf). I am using the EPANET simulator because the WNTR simulator cannot handle water quality.
Represent some results - the criteria that will be used for optimisation are not defined, yet.

In order to run the script, installing wntr (https://wntr.readthedocs.io/en/latest/installation.html) should be enough, everything else is in the folder where the MAIN script is located.
