#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <experimental/filesystem>
#include "TinyEXIF.h"

using namespace cv;
using namespace std;
using namespace Eigen;
namespace fs = experimental::filesystem;

double Get_exposure_time(string path)  // Read exif. Parsing metadata inside jpg/jpeg, https://github.com/cdcseacave/TinyEXIF
{
	std::ifstream file(path, std::ifstream::in | std::ifstream::binary);
	file.seekg(0, std::ios::end);
	std::streampos length = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<uint8_t> data(length);
	file.read((char*)data.data(), length);
	TinyEXIF::EXIFInfo imageEXIF(data.data(), length);
	if (imageEXIF.Fields) {
		std::cout << "Image exposure time in seconds: " << imageEXIF.ExposureTime<<endl;
		return imageEXIF.ExposureTime;
	}
	else {
		return 0;
	}
}

int main(int argc, char** argv )
{
	string path = "my_hdr_img";
	for (auto & p : fs::directory_iterator(path)) // Take all files inside folder path
	{
		std::cout << p << std::endl;
		Mat image;
		image = imread(p.path().string(), CV_LOAD_IMAGE_COLOR); // Read the file
		if (!image.data) // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
		}
		else {
			double exposure_time = Get_exposure_time(p.path().string()); // Get exposure time of image
			imshow("image", image);
			waitKey(0);
		}
	}
	waitKey(0);
    return 0;
}