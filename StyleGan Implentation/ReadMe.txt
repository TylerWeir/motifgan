

Gan explination

Gan Network two important componets

Discriminator 
	- Tries to see what is a real sample and a fake sample
Generator
	- Generates Fake image from Random noise
	
Goal is to make the fake as good as the real
	We want the discrimnator and generator to improve
	
Transposed convulational layer used Nearest-Neighbor sampling
	Copies nearest pixel value
Bi-linear sampling: use all nearby values to calculate the pixel value using linear intepolations. 
	
Why StyleGan
	StyleGan is open source project from NVIDIA
	Architecture different from previous
		1) Progressive growing (Low Res to High Res)
		2) noise mapping (Bi-linear scaling)
		3) adaptive instance normalization 
		
		
Notes For Dev

Goal Try to implement StyleGan with our dataset
