# Phase Experiment

During the week of 3/1 we worked on performing an experiment between two phase reconstruction methods.
We compared the outputs of two (nearly identical) GANs; one used the Griffin Lim algorithm, the other
generated the phase itself in another channel. 

### Result
We found the GAN working on just the magnitude based samples produced better results more quickly and 
more reliably than the GAN working with a phase channel. We will move forward with the Griffin Lim 
approach. 