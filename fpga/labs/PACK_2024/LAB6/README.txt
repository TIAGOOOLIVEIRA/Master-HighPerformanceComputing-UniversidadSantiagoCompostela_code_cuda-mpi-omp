Try first 1D_conv.cpp and later 1D_conv_v2.cpp.

The main issue with 1D_conv is that we try to write five words at the same time but the data array is stored in a BRAM with limited read/write ports. So we can't achieve II=1.

One solution is to reuse the data previously read implementing shift register/buffer (see 1D_conv_v2).
