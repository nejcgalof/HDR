#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <random>
#include <experimental/filesystem>
#include "TinyEXIF.h"
#include "opencv2/photo.hpp"

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

int* Get_random_pixels(Mat img, int num) {
	int* pixels = new int[num * 2];
	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uni_width(0, img.size().width); // guaranteed unbiased
	std::uniform_int_distribution<int> uni_height(0, img.size().height); // guaranteed unbiased
	for (int i = 0; i < num; i+=2) {
		int x = uni_width(rng);
		int y = uni_height(rng);
		pixels[i] = x;
		pixels[i + 1] = y;
	}
	return pixels;
}

int* Create_weights() {
	int* weights = new int[256];
	int max = 255;
	int min = 0;
	int mid = (max - min) / 2;
	for (int i = 0; i < 256; i++) {
		if (i < mid) {
			weights[i] = i - min;
		}
		else {
			weights[i] = max - i;
		}
	}
	return weights;
}

float weight(int bit8)
{
	//if (bit8 < 128) return bit8 + 1.f;
	//else return 256.f - bit8;
	if (bit8 < 128) return bit8;
	else return 255.f - bit8;
}


int main(int argc, char** argv )
{
	string path = "HDR_imgs/HDR_imgs/slike/scene";
	Vector<Mat> images;
	Vector<double> exposure_times;
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
			images.push_back(image);
			exposure_times.push_back(exposure_time);
			//imshow("image", image);
			//waitKey(0);
		}
	}
	int rows = images[0].size().height;
	int cols = images[0].size().width;
	int sample_x[100];
	int sample_y[100];

	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uni_width(0, images[0].size().width-1); // guaranteed unbiased
	std::uniform_int_distribution<int> uni_height(0, images[0].size().height-1); // guaranteed unbiased
	for (int i = 0; i < 100; i ++) {
		int x = uni_width(rng);
		int y = uni_height(rng);
		sample_x[i] = x;
		sample_y[i] = y;
	}

	// SVD for Finding Response Function
	float Im[3][256];
	for (int cBGR = 0; cBGR < 3; cBGR++)
	{
		Mat A = Mat::zeros(images.size() * 100 + 1 + 254, 256 + 100, CV_32F);
		Mat B = Mat::zeros(images.size() * 100 + 1 + 254, 1, CV_32F);

		int rowcnt = 0;

		for (int s = 0; s < 100; s++)
		{
			for (int t = 0; t < images.size(); t++)
			{
				int bit8 = images[t].at<Vec3b>(sample_y[s], sample_x[s])[cBGR];
				float w = weight(bit8);

				A.at<float>(rowcnt, bit8) = w;
				A.at<float>(rowcnt, 256 + s) = -w;
				B.at<float>(rowcnt, 0) = w * log(exposure_times[t]);
				rowcnt++;
			}
		}

		A.at<float>(rowcnt, 128) = 1;
		rowcnt++;

		for (int i = 1; i <= 254; i++)
		{
			float lambda = 10; // Using OpenCV default
			A.at<float>(rowcnt, i) = -2 * lambda * weight(i);
			A.at<float>(rowcnt, i - 1) = lambda * weight(i);
			A.at<float>(rowcnt, i + 1) = lambda * weight(i);
			rowcnt++;
		}

		Mat x_star;
		solve(A, B, x_star, DECOMP_SVD); // Pseudo Inverse

		for (int i = 0; i < 256; i++)
			Im[cBGR][i] = x_star.at<float>(i);

		// Final HDR
		Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);


		for (int cBGR = 0; cBGR < 3; cBGR++)
		{
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					float logIsum = 0;

					for (int t = 0; t < images.size(); t++)
					{
						
						int bit8 = images[t].at<Vec3b>(r, c)(cBGR);
						//if (bit8 > 10 && bit8 < 245) {
							logIsum += Im[cBGR][bit8] - log(exposure_times[t]);
						//}
						
					}

					HDR.at<Vec3f>(r, c)(cBGR) = log(exp(logIsum / images.size()));
				}
			}
		}
		Mat bgr[3];   //destination array
		split(HDR, bgr);

		for (int cBGR = 0; cBGR < 3; cBGR++)
		{
			double min, max;
			cv::minMaxLoc(bgr[cBGR], &min, &max);
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					double Zi = HDR.at<Vec3f>(r, c)(cBGR);
					double norm = (Zi - min) / (max - min);
					HDR.at<Vec3f>(r, c)(cBGR) = norm;
				}
			}
		}
		imshow("HDR", HDR);
		HDR.convertTo(HDR, CV_8UC3, 255.0);
		imwrite("resultHDR.jpg", HDR);
	}

	waitKey(0);
    return 0;
}