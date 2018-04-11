#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <random>
#include <experimental/filesystem>
#include "TinyEXIF.h"
#include "opencv2/photo.hpp"

using namespace cv;
using namespace std;
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

int* Create_weights() 
{
	int* weights = new int[256];
	int max = 255;
	int min = 0;
	int mid = (max + min) / 2;
	for (int i = 0; i < 256; i++) {
		if (i <= mid) {
			weights[i] = i - min;
		}
		else {
			weights[i] = max - i;
		}
	}
	return weights;
}

Mat Reconstruction(Vector<Mat> images, Vector<double> exposure_times, float g[3][256])
{
	Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);
	for (int cBGR = 0; cBGR < 3; cBGR++)
	{
		for (int r = 0; r < images[0].size().height; r++)
		{
			for (int c = 0; c < images[0].size().width; c++)
			{
				float logIsum = 0;
				int cnt = 0;
				for (int t = 0; t < images.size(); t++)
				{
					int Z = images[t].at<Vec3b>(r, c)(cBGR);
					if (Z > 10 && Z < 245) { // don't use pixels, which are too much or too low exposed
						logIsum += g[cBGR][Z] - log(exposure_times[t]);
						cnt++;
					}
				}
				HDR.at<Vec3f>(r, c)(cBGR) = log(exp(logIsum / cnt));
			}
		}
	}
	return HDR;
}

Mat Normalization(Mat HDR)
{
	Mat bgr[3];   //destination array
	split(HDR, bgr);

	for (int cBGR = 0; cBGR < 3; cBGR++)
	{
		double min, max;
		cv::minMaxLoc(bgr[cBGR], &min, &max);
		for (int r = 0; r < HDR.size().height; r++)
		{
			for (int c = 0; c < HDR.size().width; c++)
			{
				double Zi = HDR.at<Vec3f>(r, c)(cBGR);
				double norm = (Zi - min) / (max - min);
				HDR.at<Vec3f>(r, c)(cBGR) = norm;
			}
		}
	}
	HDR.convertTo(HDR, CV_8UC3, 255.0);
	return HDR;
}

void gsolve(Vector<Mat> images, Vector<double> exposure_times, int* weights, int* sample_x, int* sample_y, float g[3][256], float lambda = 10)
{
	for (int cBGR = 0; cBGR < 3; cBGR++)
	{
		Mat A = Mat::zeros(images.size() * 100 + 1 + 256, 256 + 100, CV_32F);
		Mat b = Mat::zeros(images.size() * 100 + 1 + 256, 1, CV_32F);

		// Include the data-fitting equations
		int k = 0;
		for (int i = 0; i < 100; i++)
		{
			for (int j = 0; j < images.size(); j++)
			{
				int pixel = images[j].at<Vec3b>(sample_y[i], sample_x[i])[cBGR];
				float w = weights[pixel];

				A.at<float>(k, pixel) = w;
				A.at<float>(k, 256 + i) = -w;
				b.at<float>(k, 0) = w * log(exposure_times[j]);
				k++;
			}
		}

		// Fix the curve by setting its middle value to 0
		A.at<float>(k, 128) = 1;
		k++;

		// Include the smoothness equations
		for (int i = 1; i <= 254; i++)
		{
			A.at<float>(k, i) = -2 * lambda * weights[i];
			A.at<float>(k, i - 1) = lambda * weights[i];
			A.at<float>(k, i + 1) = lambda * weights[i];
			k++;
		}

		// Solve the system using SVD
		Mat x;
		solve(A, b, x, DECOMP_SVD); // Pseudo Inverse

		for (int i = 0; i < 256; i++) {
			g[cBGR][i] = x.at<float>(i);
		}
	}
}

int main(int argc, char** argv )
{
	// Get images and exposure times
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
		}
	}
	
	// Random 100 pixels locations inside image
	int sample_x[100];
	int sample_y[100];
	random_device rd;
	mt19937 rng(rd());
	uniform_int_distribution<int> uni_width(0, images[0].size().width-1); // guaranteed unbiased
	uniform_int_distribution<int> uni_height(0, images[0].size().height-1); // guaranteed unbiased
	for (int i = 0; i < 100; i ++) {
		int x = uni_width(rng);
		int y = uni_height(rng);
		sample_x[i] = x;
		sample_y[i] = y;
	}

	// Generate weights - weigthing function values
	int* weights = Create_weights();

	// SVD for Finding Response Function
	float g[3][256]; // is the log exposure corresponding to pixel value
	gsolve(images, exposure_times, weights, sample_x, sample_y, g);

	// Reconstruction
	Mat HDR = Reconstruction(images, exposure_times, g);

	// Normalization
	HDR = Normalization(HDR);

	imwrite("resultHDR.jpg", HDR);
	imshow("HDR", HDR);
	waitKey(0);
    return 0;
}