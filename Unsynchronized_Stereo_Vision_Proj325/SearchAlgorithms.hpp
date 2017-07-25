#ifndef SearchAlgorithms_HPP
#define SearchAlgorithms_HPP

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mutex>
#include <future>
#include <stdio.h> 
#include <math.h>
#include "Match.hpp"
#include "DistanceCalculator.hpp"
#include "SearchAlgorithms.hpp""

//Namespace list
using namespace cv;
using namespace std;
using namespace std::chrono;

/*
ColourSearchParameters
Stores int values required for HSV colour thresholding

int iLowHue, iLowSaturation, iLowValue,
iHighHue, iHighSaturation, iHighValue,
iLowHue2, iHighHue2;
*/
struct ColourSearchParameters {
	//SliderValue
	int iLowHue, iLowSaturation, iLowValue,
		iHighHue, iHighSaturation, iHighValue,
		iLowHue2, iHighHue2;
};

void MorphilogicalFilter(Mat ThesholdImage);

void ABSDiffSearch(Mat *Gray, Mat& ThesholdImage, Mat *ImportPrev, Mat& ExportPrev);

void ColourSearch(Mat *HSVImage, Mat& ThesholdImage, ColourSearchParameters *SliderValue);

int CannySearch(bool CameraSide, Mat *ImportGrayThisCamera, std::vector<std::vector<Point>> *ImportCannyUsefulContoursOtherCamera,
	std::vector<Point2f> *ImportVectorCenter_pointOtherCamera, std::vector<std::vector<Point>>& ExportCannyUsefulContoursThisCamera,
	 std::vector<Point2f>& ExportVectorCenter_pointThisCamera, Mat& ExportContourOverlay, Mat& ExportDebugCannyImg);

#endif /* SearchAlgorithms_HPP */