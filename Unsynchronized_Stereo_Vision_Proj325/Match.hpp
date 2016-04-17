#ifndef Match_HPP
#define Match_HPP
#include <string>

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

class Match{
     
    public:
    Match(unsigned int LeftIndex, unsigned int  RightIndex,double MatchValue);       //constructor of the class
	unsigned int LeftIndex;
	unsigned int RightIndex;
	double MatchValue;

};


//class CenterPointDataParameters{
//	public:
//	CenterPointDataParameters(std::vector<Point2f> ImportedVectorCenter_point,
//		std::vector<Point2f> ImportedOldVectorCenter_point, std::vector<Point2f> ImportedOlderVectorCenter_point,
//		std::vector <Point3i> InterframeMatchIndexesCompleteR);
//	CenterPointDataParameters& CenterPointDataParameters::pointer(std::vector<Point2f> VectorCenter_point,
//		std::vector<Point2f> OldVectorCenter_point, std::vector<Point2f> OlderVectorCenter_point,
//		std::vector <Point3i> InterframeMatchIndexesComplete);
//
//	std::vector<Point2f> VectorCenter_point;
//	std::vector<Point2f> OldVectorCenter_point;
//	std::vector<Point2f> OlderVectorCenter_point;
//	std::vector <Point3i> InterframeMatchIndexesComplete;
//};
;

//class {
//pulbic:

//	std::vector<Point2f> ExportedVectorCenter_pointL;
//	std::vector<Point2f> ExportedOldVectorCenter_pointL;
//	std::vector<Point2f> ExportedOlderVectorCenter_pointL;
//	std::vector <Point3i> ExportedInterframeMatchIndexesComplete;
//};

#endif /* Match_HPP */