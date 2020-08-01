from qnn_param_reader import QNNParamReader
from qnn_mem_process import QNNLayerMemProcess
import json
import os

# conv       0      1   2       3   4   5
w_bit   =   [4,     4,  4,      4,  4,   8]
in_bit  =   [8,     4,  4,      4,  4,   4]
out_bit =   [4,     4,  4,      4,  4,  32]
l_shift =   [6,     6,  6,      6,  6,   6]
simd    =   [3,     16,  16,    16, 8,   8]
pe      =   [16,    8,   8,     4,  2,   2]
    


if __name__ == "__main__":

    target_dir_hls_param = 'param/hls/'
    if not os.path.exists(target_dir_hls_param):
        os.makedirs(target_dir_hls_param)
    
    hls_param_file = open(target_dir_hls_param + 'param.h', 'w')
    hls_config_file = open(target_dir_hls_param + 'config.h', 'w')

    config_file = open('config.json', 'r', encoding='utf-8')
    config = json.load(config_file)
    reader = QNNParamReader('XianKing_4w4a.npz')

    # conv_0 - 4
    for i in range(0, 5):
        processer = QNNLayerMemProcess('conv_' + str(i), reader, config, w_bit=w_bit[i], in_bit=in_bit[i], out_bit=out_bit[i], l_shift=l_shift[i], pe=pe[i], simd=simd[i])
        w, inc, bias = processer.conv()
        param_str = processer.layer_param_to_init_str(w, inc, bias)
        config_str = processer.conv_config_str()
        hls_param_file.write(param_str)
        hls_config_file.write(config_str)

    processer = QNNLayerMemProcess('conv_' + str(5), reader, config, w_bit=w_bit[5], in_bit=in_bit[5],
                                   out_bit=out_bit[5], l_shift=l_shift[5], pe=pe[5], simd=simd[5])
    w = processer.last_conv()
    param_str = processer.last_layer_param_to_init_str(w)
    config_str = processer.last_conv_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    hls_param_file.close()
    hls_config_file.close()
    
