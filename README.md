# Image_only_model
### This version uses new pooling and new dense_ss.
### The new dense_ss is modified by me. (can adjust reuse factor)
### I also compare with Dylan's project to adjust the reuse factor.
### The synthesis time will be a liitle longer. ( 1hour_10mins -> 1hour_50mins )
### The upsamping layer & normalized layer & leaky_relu layer all have been modified.
### I uncommeted the pragma on " layer_in ". As for 9th conv2d, I still commeted it.
### The latency reduced a lot!!! However, the number of DSP is scaring! > <
### Another report is adjusting the precision to ap_fixed<16,6>, and the number of DSP reduced to 1/4.
### The COSIM result is correct!!
