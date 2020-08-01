#include <ap_int.h>
#include <hls_video.h>
#include <hls_stream.h>
#include <iostream>
#include <fstream>
#include "stream_tools.h"
using namespace std;

void king_net(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps);

void load_data(const char *path, char *ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}

void write_data(const char *path, char *ptr, unsigned int size)
{
    std::ofstream f(path, std::ios::out | std::ios::binary);
    if (!f)
    {
        std::cout << "write no such file,please check the file name!/n";
        exit(0);
    }
    f.write(ptr, size);
    f.close();
}

int main(int argc, char const *argv[])
{
    uint8_t img[128][128][3];
    load_data("./test_data/test_8.bin", (char *) img, sizeof(img));

    uint8_t * data = (uint8_t *) img;
    const int data_points_per_line = 8;
    const int nums_line_pre_img = 128 * 128 * 3 / 8;

    hls::stream<my_ap_axis> input_stream("input stream");
	for (unsigned int i = 0; i < nums_line_pre_img; i++) {
		my_ap_axis temp;
		for (unsigned int j = 0; j < data_points_per_line; j++) {
			temp.data( 8*(j+1)-1, 8*j ) = data[i * data_points_per_line + j];
		}
		input_stream.write(temp);
	}
	cout << "input size :" << input_stream.size() << endl;
    cout << "start ..... " << endl;
    hls::stream<my_ap_axis> output_stream("output stream");
    king_net(input_stream, output_stream, 1);

    ap_uint<64> outputBuffer[5];
    ap_uint<32> Buffer[10];

	for (unsigned int i = 0; i < 5; i++) {
		outputBuffer[i] = output_stream.read().data;
		Buffer[2*i]=outputBuffer[i](31,0);
		Buffer[2*i+1]=outputBuffer[i](63,32);
	}

	unsigned int prediction = -1;
	int max = -(1<<30);

	for (unsigned int i = 0; i < 10; i++) {
		int temp = Buffer[i].to_int();
		cout << temp << endl;
		if (temp > max){
			max = temp;
			prediction = i;
		}
	}

	cout << "prediction: " << prediction << endl;

    
    return 0;
}
