#include <opencv2/opencv.hpp> 
#include <opencv2/calib3d/calib3d.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <stdio.h> 
#include <iostream> 
// Задаем размеры шахматной доски 

int CHECKERBOARD[2]{ 6,9 };
int main()
{
	// Для хранения трехмерных точек шахматной доски 
	std::vector<std::vector<cv::Point3f> > objpoints;
	// Для хранения двумерных точек шахматной доски 
	std::vector<std::vector<cv::Point2f> > imgpoints;
	// Определяем координаты точек в мировой системе координат 
	std::vector<cv::Point3f> objp;
	for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
	{
		for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
			objp.push_back(cv::Point3f(j, i, 0));
	}
	// Находим папку где хранятся изображения 
	std::vector<cv::String> images;
	std::string path = "./images/*.jpg";
	cv::glob(path, images);
	cv::Mat frame, gray;
	std::vector<cv::Point2f> corner_pts;
	bool success;
	// Запускаем цикл пока не пройдем все изображения 
	for (int i{ 0 }; i < images.size(); i++)
	{
		frame = cv::imread(images[i]);
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		// Ищем углы шахматной доски 
		// Если найдены все углы то success = true 
		success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
		/*
		* Если найдено ожидаемое количество углов
		* Вычисляются координаты в пикселях
		* Отображается изображения с отмеченными на ней углами
		*/
		if (success)
		{
			cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
			cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);
			cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
			objpoints.push_back(objp);
			imgpoints.push_back(corner_pts);

		}
		cv::imshow("Image", frame);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();
	cv::Mat cameraMatrix, distCoeffs, R, T;
	cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
	std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
	std::cout << "distCoeffs : " << distCoeffs << std::endl;
	std::cout << "Rotation vector : " << R << std::endl;
	std::cout << "Translation vector : " << T << std::endl;

	cv::FileStorage fs;
	if (fs.open("camera.yml", cv::FileStorage::WRITE))
	{
		fs << "CMat" << cameraMatrix;
		fs << "DMat" << distCoeffs;
		fs << "R" << R;
		fs << "T" << T;
	}

	return 0;
}