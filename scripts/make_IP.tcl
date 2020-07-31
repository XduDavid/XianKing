############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project KingNet
set_top king_net
add_files config.h
add_files conv2d.h
add_files function.h
add_files kingnet.cpp
add_files matrix_vector_unit.h
add_files param.h
add_files pool2d.h
add_files sliding_window_unit.h
add_files stream_tools.h
add_files -tb test.cpp
add_files -tb test_data
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e} -tool vivado
create_clock -period 5 -name default

csim_design -clean
csynth_design
#cosim_design
export_design -rtl verilog -format ip_catalog
