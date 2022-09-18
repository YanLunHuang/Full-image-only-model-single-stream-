# Image_only_model
### precision ap_fixed<16,6>
### The difference between this version and " 8_31_array_of_stream_16_6 " is that it can change the data type automatically.
### When the channel number >= 64, it will use single stream as data type.
### I commented the pragma INLINE above dense_large so that it will not increase extra LUT. (compared to branch 9_14_auto_change)
### The synthesis time is about 1.5 hours.
### Both of the resource utilization and latency are normal.
