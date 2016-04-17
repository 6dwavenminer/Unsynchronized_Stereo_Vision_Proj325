#include "Match.hpp"
#include <opencv2/opencv.hpp>


#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <fstream>

#include "opencv2/imgcodecs.hpp"
#include <windows.h>
#include <process.h>
#include <iostream>

using namespace cv;
using namespace std;

//Constructor
Match::Match(unsigned int LeftIndex,unsigned int RightIndex,double MatchValue){
	this->LeftIndex = LeftIndex;
	this->RightIndex = RightIndex;
	this->MatchValue = MatchValue;

}


//CenterPointDataParameters::CenterPointDataParameters(std::vector<Point2f> VectorCenter_point,
//	std::vector<Point2f> OldVectorCenter_point,std::vector<Point2f> OlderVectorCenter_point,
//	std::vector <Point3i> InterframeMatchIndexesComplete) {
//	//this->VectorCenter_point=VectorCenter_point;
//	//this->OldVectorCenter_point=OldVectorCenter_point;
//	//this->OlderVectorCenter_point=OlderVectorCenter_point;
//	//this->InterframeMatchIndexesComplete=InterframeMatchIndexesComplete;
//}
//CenterPointDataParameters& CenterPointDataParameters::pointer(std::vector<Point2f> VectorCenter_point,
//	std::vector<Point2f> OldVectorCenter_point, std::vector<Point2f> OlderVectorCenter_point,
//	std::vector <Point3i> InterframeMatchIndexesComplete) {
//};
;
