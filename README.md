# Image_only_model
### precision ap_fixed<32,16>
### The difference between this version and " 8_31_array_of_stream_16_6 " is that it can change the data type automatically.
### When the channel number >= 64, it will use single stream as data type.
### I commented the pragma INLINE above dense_large so that it will not increase extra LUT. (compared to branch 9_14_auto_change)
### The branch has improved all of the layers in image-only model so the performance will be better than " 9_18_auto_change_final ".
### The synthesis time is about 1.2 hours.
### The copy_data & print_result function can auto change the data type.
### Both of the resource utilization and latency are normal.
