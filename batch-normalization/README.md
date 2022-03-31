### DATE
3/21/2022

### Description
Adding batch normalization layers and switched to relu as per recomendation in the Alec Radfords guidelines for building stable convolutional GANS (see pg 598 of Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow).  Experimented with filtering the noise out of GAN output.

### Results
During training, the loss functions of both the Generator and the Discriminator were slow and steady.  I trained the model for 800 epochs total to see if further training improved output. It did not have a noticeable effect after ~300 epochs.  Batch norm seems to have improved output marginally, though it is difficult to attribute specifically.  Perhaps the best thing to come from this round of exploration is the idea of filtering the GAN output.  Currently, output has lots of extra noise which results in poor output quality.  I found that removing all array values less than 0.1 removes the noise and improves the output sound significantly.