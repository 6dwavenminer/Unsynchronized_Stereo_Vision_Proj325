#ifndef DistanceCalculator_HPP
#define DistanceCalculator_HPP

//#include <cstdio>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <stdio.h> 
#include <math.h>


//Namespace list
using namespace cv;
using namespace std;
using namespace std::chrono;


#define LeftCam true
#define RightCam false

#define XYFOVangle 70
#define ZYFOVangle 70
#define XPixelDimensions 640
#define YPixelDimensions 480
#define CameraDistcm 20.16 
#define PI 3.14159265

//Global control variables
/*
*/
extern bool CoordinateDisplay;


double deg2rad(double deg);

double rad2deg(double rad);

void MovingObjectDistanceCalculator(bool CameraSide, std::chrono::steady_clock::time_point ImgTimeStampThisCamera, std::vector<Point2f> VectorCenter_pointThisCamera,
	std::vector<Point2f> VectorCenter_pointOtherCamera,
	std::vector<Point2f> OldVectorCenter_pointOtherCamera,
	std::vector<Point2f> OlderVectorCenter_pointOtherCamera,
	std::vector<Point2f> InterpolatedVectorCenter_pointOtherCamera,
	std::vector<Point3i> InterframeMatchIndexesCompleteOtherCamera,
	std::chrono::steady_clock::time_point ImgTimeStampOtherCamera,
	std::chrono::steady_clock::time_point OldImgTimeStampOtherCamera,
	std::chrono::steady_clock::time_point OlderImgTimeStampOtherCamera,
	std::vector<double> &dist);

void CooridinatePositionCalculator(bool CameraSide, std::vector<double> dist, std::vector<Point2f> VectorCenter_pointThisCamera, vector<Point3d> &PoscmFromReferencePointVector);


#endif /* DistanceCalculator_HPP */