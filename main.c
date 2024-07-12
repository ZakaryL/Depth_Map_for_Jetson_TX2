#include <iostream> 
#include <string> 
#include <vector> 
#include <unordered_map> 
#include <stdlib.h> 
#include <chrono> 
#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/videoio.hpp> 
#include <opencv2/video.hpp> 
#include <opencv2/cudaarithm.hpp> 
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudawarping.hpp> 
#include <opencv2/cudaoptflow.hpp> 
#include <opencv2/aruco.hpp> 

void optical_flow(float f, cv::Mat K, cv::Mat dist)
{

	cv::VideoCapture capture("/dev/video0");
	int width, height;

	cv::Mat c_Frame, r_Frame, dispMap;
	cv::cuda::GpuMat gc_Frame, gp_Frame, g_flow;
	cv::cuda::GpuMat g_Magn;

	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);

	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners;
	std::vector<cv::Vec3d> rvecs, tvecs;

	if (!capture.isOpened())
	{
		std::cout << "Unable to open file!" << std::endl;
		return;
	}
	else
	{
		width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
		height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
		std::cout << "Resolution: " << width << "*" << height <<
			std::endl;
	}

	cv::Point2i resolution(width, height);

	capture >> c_Frame;
	cv::resize(c_Frame, c_Frame, cv::Size(resolution), 0, 0, cv::INTER_LINEAR);

	cv::cvtColor(c_Frame, c_Frame, cv::COLOR_BGR2GRAY);

	gp_Frame.upload(c_Frame);

	cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> ptr_calc = cv::cuda::OpticalFlowDual_TVL1::create(0.25, 0.15, 0.3, 4, 4, 0.5, 150, 0.6, 0.0, false);

	while (true)
	{
		capture >> c_Frame;
		if (c_Frame.empty())
			break;
		gc_Frame.upload(c_Frame);
		cv::cuda::resize(gc_Frame, gc_Frame, cv::Size(resolution), 0, 0, cv::INTER_LINEAR);
	    cv::cuda::cvtColor(gc_Frame, gc_Frame, cv::COLOR_BGR2GRAY);
		ptr_calc->calc(gp_Frame, gc_Frame, g_flow);

		cv::cuda::magnitude(g_flow, g_Magn);

		g_Magn.download(dispMap);

		cv::Mat depthMap = cv::Mat::zeros(dispMap.size(), CV_32FC1);
		for (int y = 0; y < dispMap.rows; ++y)
		{
			for (int x = 0; x < dispMap.cols; ++x)
			{
				float disp = dispMap.at<float>(y, x);
				if (disp > 1)
				{
					float depth = 0.1 * f / disp;
					depthMap.at<float>(y, x) = depth;
				}
				else
				{
					depthMap.at<float>(y, x) = 0;
				}
			}
		}

		cv::aruco::detectMarkers(c_Frame, dictionary, corners, ids);
		if (ids.size() > 0)
		{
			cv::aruco::estimatePoseSingleMarkers(corners, 0.15, K, dist, rvecs, tvecs);

			for (int i = 0; i < ids.size(); i++)
			{
				double cent_x = (corners[i][0].x + corners[i][1].x + corners[i][2].x + corners[i][3].x) / 4;
				double cent_y = (corners[i][0].y + corners[i][1].y + corners[i][2].y + corners[i][3].y) / 4;

				cv::Mat R = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
				cv::Rodrigues(rvecs[i], R);
				cv::transpose(R, R);
				cv::Mat T = -R * tvecs[i];
				double d = cv::norm(T) - 0.08;
				double angle = rvecs[i][2] * 180 / CV_PI;

				char distance[50];
				double err = depthMap.at<float>(cent_x, cent_y) * 0.01;
				sprintf(distance, "Dist %d: %.3f Angle: %.3f Dist_of: % .3f", ids[i], d, angle, err); 
				cv::putText(c_Frame, distance, cv::Point(15, 15 + i * 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 0, 0), 2);
				cv::circle(c_Frame, cv::Point(cent_x, cent_y), 2, cv::Scalar(0, 0, 255), -1);
			}
		}

		cv::normalize(depthMap, depthMap, 0.0, 1.0, cv::NORM_MINMAX, -1);
		r_Frame = depthMap * 255;
		r_Frame.convertTo(r_Frame, CV_8UC1);

		gp_Frame = gc_Frame;

		imshow("original", c_Frame);
		imshow("depth", r_Frame);
		int keyboard = cv::waitKey(1);
		if (keyboard == 27)
		{
			break;
		}
		if (keyboard == 32)
		{
			cv::waitKey(0);
		}

	}

	capture.release();
	cv::destroyAllWindows();
}

int main(int argc, const char** argv)
{
	std::string method;
	cv::Mat K, Kinv, K_32f;
	cv::Mat dist, dist_32f;

	cv::FileStorage fs;
	if (fs.open("camera.yml", cv::FileStorage::READ))
	{
		fs["CMat"] >> K;
		fs["DMat"] >> dist;
	}

	K.convertTo(K_32f, CV_32FC1);

	float f = (K_32f.at<float>(0, 0) + K_32f.at<float>(1, 1)) / 2;

	optical_flow(f, K, dist);

	return 0;
}