/*
Unsynchronized_Stereo_Vision_Proj325
Author: Oliver Thurgood
Version: 0.2.6
Created on: 05/02/2016
Last edited on: 17/04/2016
*/


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
#include <thread>
#include <chrono>
#include <mutex>
#include <future>
#include "Match.hpp"

//Namespace list
using namespace cv;
using namespace std;
using namespace std::chrono;

//Mutex list
mutex mtPrev;
mutex LeftTimeStampMutex;
mutex RightTimeStampMutex;
mutex LeftContourMutex;
mutex RightContourMutex;
mutex LeftVectorCenterMutex;
mutex RightVectorCenterMutex;
mutex LeftDistanceMutex;
mutex RightDistanceMutex;

#define LeftCam true
#define RightCam false

struct absdiffFinderParameters {
	//parameters here
	Mat GrayL;
	Mat GrayR;
	Mat ThesholdImageL;
	Mat ThesholdImageR;
};

struct ColourSearchParameters {
	//SliderValue
	int iLowHue, iLowSaturation, iLowValue,
		iHighHue, iHighSaturation, iHighValue,
		iLowHue2, iHighHue2, iMorphologicalOpening,iMorphologicalClosing;
};

struct CalibrationDataParameters {
	Mat intrinsicL, distCoeffsL, intrinsicR, distCoeffsR;
	Mat RotationMat, TranslationMat, EssentailMat, FundamentalMat;
	Mat RectificationTransformMatL, RectificationTransformMatR, ProjectionMatL, ProjectionMatR, Disparity2DepthMappingMat;
	Mat map1x, map1y, map2x, map2y;
};

struct CenterPointDataParameters {
	std::vector<Point2f> VectorCenter_point;
	std::vector<Point2f> OldVectorCenter_point;
	std::vector<Point2f> OlderVectorCenter_point;
	std::vector <Point3i> InterframeMatchIndexesComplete;
	std::chrono::steady_clock::time_point ImgTimeStamp;
	std::chrono::steady_clock::time_point OldImgTimeStamp;
	std::chrono::steady_clock::time_point OlderImgTimeStamp;
};

void ABSDiffSearchOld(Mat GrayL,Mat GrayR,Mat &ThesholdImageL,Mat &ThesholdImageR, Mat &PrevL, Mat &PrevR) {
	mtPrev.lock();
	if (PrevL.empty()) {
		PrevL = GrayL;
		PrevR = GrayR;
	}
	absdiff(GrayL, PrevL, ThesholdImageL);
	absdiff(GrayR, PrevR, ThesholdImageR);

	PrevL = GrayL;
	PrevR = GrayR;
	mtPrev.unlock();

	threshold(ThesholdImageL, ThesholdImageL, 20, 255, THRESH_BINARY);
	threshold(ThesholdImageR, ThesholdImageR, 20, 255, THRESH_BINARY);

	erode(ThesholdImageL, ThesholdImageL, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
	erode(ThesholdImageR, ThesholdImageR, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
	dilate(ThesholdImageL, ThesholdImageL, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
	dilate(ThesholdImageR, ThesholdImageR, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
}

void ABSDiffSearch(Mat Gray, Mat &ThesholdImage, Mat &Prev) {
	mtPrev.lock();
	if (Prev.empty()) {
		Prev = Gray;
	}
	absdiff(Gray, Prev, ThesholdImage);

	Prev = Gray;
	mtPrev.unlock();

	threshold(ThesholdImage, ThesholdImage, 20, 255, THRESH_BINARY);

	erode(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
	dilate(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
}

void CannySearchOld(Mat* GrayL, Mat* GrayR, Mat &ThesholdImageL, Mat &ThesholdImageR, Mat &PrevL, Mat &PrevR, Mat &PrevL2, Mat &PrevR2) {
	while (1) {
		Mat BufferThesholdImageL,BufferThesholdImageR;
		if (! (*GrayL).empty()) {
			mtPrev.lock();
			if (!PrevL.empty()) {
				PrevL2 = PrevL;
			}
			if (!ThesholdImageL.empty()) {
				PrevL = BufferThesholdImageL;
			}
			if (!PrevR.empty()) {
				PrevR2 = PrevR;
			}
			if (!ThesholdImageR.empty()) {
				PrevR = BufferThesholdImageR;
			}
			Canny(*GrayL, BufferThesholdImageL, 30, 300, 3);//8
			Canny(*GrayR, BufferThesholdImageR, 30, 300, 3);//8
			if (!PrevL.empty()) {
				if (!PrevL2.empty()) {
					cv::addWeighted(BufferThesholdImageL, 1, PrevL2, 1, 0, BufferThesholdImageL);
					cv::addWeighted(BufferThesholdImageL, 1, PrevL, 1, 0, BufferThesholdImageL);
				}
			}
			if (!PrevR.empty()) {
				if (!PrevR2.empty()) {
					cv::addWeighted(BufferThesholdImageR, 1, PrevR2, 1, 0, BufferThesholdImageR);
					cv::addWeighted(BufferThesholdImageR, 1, PrevR, 1, 0, BufferThesholdImageR);
				}
			}
			mtPrev.unlock();

			dilate(BufferThesholdImageL, BufferThesholdImageL, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
			dilate(BufferThesholdImageR, BufferThesholdImageR, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

			bitwise_not(BufferThesholdImageL, BufferThesholdImageL);
			bitwise_not(BufferThesholdImageR, BufferThesholdImageR);

			threshold(BufferThesholdImageL, BufferThesholdImageL, 20, 255, THRESH_BINARY);
			threshold(BufferThesholdImageR, BufferThesholdImageR, 20, 255, THRESH_BINARY);

			ThesholdImageL = BufferThesholdImageL;
			ThesholdImageR = BufferThesholdImageR;
			waitKey(33);
		}
		this_thread::sleep_for(2s);
	}
}

void CannySearch(bool CameraSide,Mat* GrayL, Mat &ThesholdImageL,Mat &PrevL, Mat &PrevL2) {
	std::vector <Match> Matcher;
	std::vector <Match> TentativeMatch;
	std::vector <Match> DeassignedMatch;
	std::vector<Vec4i> CannyhierarchyThisCamera;
	std::vector<std::vector<Point> > CannyContoursThisCamera;
	std::vector<std::vector<Point> > CannyUsefulContoursThisCamera;
	while (1) {
		Mat BufferThesholdImageL;
		if (!(*GrayL).empty()) {
			mtPrev.lock();
			if (!PrevL.empty()) {
				PrevL2 = PrevL;
			}
			if (!ThesholdImageL.empty()) {
				PrevL = BufferThesholdImageL;
			}
			Canny(*GrayL, BufferThesholdImageL, 30, 300, 3);//8
			if (!PrevL.empty()) {
				if (!PrevL2.empty()) {
					cv::addWeighted(BufferThesholdImageL, 1, PrevL2, 1, 0, BufferThesholdImageL);
					cv::addWeighted(BufferThesholdImageL, 1, PrevL, 1, 0, BufferThesholdImageL);
				}
			}
			mtPrev.unlock();

			dilate(BufferThesholdImageL, BufferThesholdImageL, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

			bitwise_not(BufferThesholdImageL, BufferThesholdImageL);

			threshold(BufferThesholdImageL, BufferThesholdImageL, 20, 255, THRESH_BINARY);

			ThesholdImageL = BufferThesholdImageL;

			//Find contours
			findContours(ThesholdImageThisCamera, CannyContoursThisCamera, CannyhierarchyThisCamera, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
			FindUsefulContours(CannyContoursThisCamera, CannyUsefulContoursThisCamera);
			//End finding contours
			//Data trade
			if (CameraSide == LeftCam) {

			}else{

			}
			//End data trade
			GenerateMatchingList(CannyUsefulContoursThisCamera, CannyUsefulContoursOtherCamera, Matcher);
			//Matcher
			ResolveMatchList(Matcher, TentativeMatch);
			//waitKey(30);
		}
		this_thread::sleep_for(2s);
	}
}

void ColourSearchOld(Mat HSVImageL, Mat HSVImageR, Mat &ThesholdImageL, Mat &ThesholdImageR,ColourSearchParameters SliderValue) {
	Mat ThesholdImageInvertedHueRangeL;
	Mat ThesholdImageInvertedHueRangeR;

	//Threshold Image
	inRange(HSVImageL, Scalar(SliderValue.iLowHue, SliderValue.iLowSaturation, SliderValue.iLowValue), Scalar(SliderValue.iHighHue, SliderValue.iHighSaturation, SliderValue.iHighValue), ThesholdImageL);	//Hue,Saturation,Value
	inRange(HSVImageL, Scalar(SliderValue.iLowHue2, SliderValue.iLowSaturation, SliderValue.iLowValue), Scalar(SliderValue.iHighHue2, SliderValue.iHighSaturation, SliderValue.iHighValue), ThesholdImageInvertedHueRangeL);	//Hue,Saturation,Value
	addWeighted(ThesholdImageL, 1, ThesholdImageInvertedHueRangeL, 1, 0, ThesholdImageL);

	//Threshold Image
	inRange(HSVImageR, Scalar(SliderValue.iLowHue, SliderValue.iLowSaturation, SliderValue.iLowValue), Scalar(SliderValue.iHighHue, SliderValue.iHighSaturation, SliderValue.iHighValue), ThesholdImageR);	//Hue,Saturation,Value
	inRange(HSVImageR, Scalar(SliderValue.iLowHue2, SliderValue.iLowSaturation, SliderValue.iLowValue), Scalar(SliderValue.iHighHue2, SliderValue.iHighSaturation, SliderValue.iHighValue), ThesholdImageInvertedHueRangeR);	//Hue,Saturation,Value
	addWeighted(ThesholdImageR, 1, ThesholdImageInvertedHueRangeR, 1, 0, ThesholdImageR);

	erode(ThesholdImageL, ThesholdImageL, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	erode(ThesholdImageR, ThesholdImageR, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	dilate(ThesholdImageL, ThesholdImageL, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	dilate(ThesholdImageR, ThesholdImageR, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
}

void ColourSearch(Mat HSVImage, Mat &ThesholdImage, ColourSearchParameters SliderValue) {
	Mat ThesholdImageInvertedHueRange;

	//Threshold Image
	inRange(HSVImage, Scalar(SliderValue.iLowHue, SliderValue.iLowSaturation, SliderValue.iLowValue), Scalar(SliderValue.iHighHue, SliderValue.iHighSaturation, SliderValue.iHighValue), ThesholdImage);	//Hue,Saturation,Value
	inRange(HSVImage, Scalar(SliderValue.iLowHue2, SliderValue.iLowSaturation, SliderValue.iLowValue), Scalar(SliderValue.iHighHue2, SliderValue.iHighSaturation, SliderValue.iHighValue), ThesholdImageInvertedHueRange);	//Hue,Saturation,Value
	addWeighted(ThesholdImage, 1, ThesholdImageInvertedHueRange, 1, 0, ThesholdImage);


	erode(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
	dilate(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
}

void LoadCalibrationData(CalibrationDataParameters &CalibrationData) {
	string filename;
	filename = "C:/Users/Oliver/Documents/Work/University Work/Uni 3rd Year/Proj325/StereoCalibration/StereoCalibration4r3.xml";

	FileStorage fs(filename, FileStorage::READ);
	fs.open(filename, FileStorage::READ);
	fs["intrinsicL"] >> CalibrationData.intrinsicL;
	fs["intrinsicR"] >> CalibrationData.intrinsicR;
	fs["distCoeffsL"] >> CalibrationData.distCoeffsL;
	fs["distCoeffsR"] >> CalibrationData.distCoeffsR;
	fs["RotationMat"] >> CalibrationData.RotationMat;
	fs["TranslationMat"] >> CalibrationData.TranslationMat;
	fs["EssentailMat"] >> CalibrationData.EssentailMat;
	fs["FundamentalMat"] >> CalibrationData.FundamentalMat;
	fs["RectificationTransformMatL"] >> CalibrationData.RectificationTransformMatL;
	fs["RectificationTransformMatR"] >> CalibrationData.RectificationTransformMatR;
	fs["ProjectionMatL"] >> CalibrationData.ProjectionMatL;
	fs["ProjectionMatR"] >> CalibrationData.ProjectionMatR;
	fs["Disparity2DepthMappingMat"] >> CalibrationData.Disparity2DepthMappingMat;
	fs.release();
}

void CalibrateLeftImage(Mat SrcImgL, CalibrationDataParameters CalibrationData, Mat &CalibratedImgL) {
	initUndistortRectifyMap(CalibrationData.intrinsicL, CalibrationData.distCoeffsL, CalibrationData.RectificationTransformMatL, CalibrationData.ProjectionMatL, SrcImgL.size(), CV_16SC2, CalibrationData.map1x, CalibrationData.map1y);
	remap(SrcImgL, CalibratedImgL, CalibrationData.map1x, CalibrationData.map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
}

void CalibrateRightImage(Mat SrcImgR, CalibrationDataParameters CalibrationData, Mat &CalibratedImgR) {
	initUndistortRectifyMap(CalibrationData.intrinsicR, CalibrationData.distCoeffsR, CalibrationData.RectificationTransformMatR, CalibrationData.ProjectionMatR, SrcImgR.size(), CV_16SC2, CalibrationData.map2x, CalibrationData.map2y);
	remap(SrcImgR, CalibratedImgR, CalibrationData.map2x, CalibrationData.map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
}

void LightingCorrection(Mat HSVImage, vector<Mat> &HSVChannels,Mat &CalibratedImg) {
	//vector<Mat> HSVChannels; cant be declared inside this function due to an OpenCV bug which fails to properally deconstruct it.
	split(HSVImage, HSVChannels);					//Split HSV into its constituant channels
	equalizeHist(HSVChannels[2], HSVChannels[2]);	//HSV value correction using histogram equalization
	merge(HSVChannels, HSVImage);					//Merge the value back into normal HSV format
	cvtColor(HSVImage, CalibratedImg, CV_HSV2BGR);	//Convert the HSV image back to BGR format
}

void FindUsefulContours(std::vector<std::vector<Point> > &Contours,std::vector<std::vector<Point> > &UsefulContours) {
	UsefulContours = Contours;
	if (Contours.size() != 0) {
		int iIndex = 0;
		int iUsefulIndex = 0;
		while ((unsigned)iIndex < Contours.size()) {
			if (Contours[iIndex].size() > 30) {
				//Simplifies the countour
				approxPolyDP(Mat(Contours[iIndex]), Contours[iUsefulIndex], 3, true);
				(UsefulContours)[iUsefulIndex] = Contours[iUsefulIndex];
				iUsefulIndex++;
			}
			iIndex++;
		}
		(UsefulContours).resize(iUsefulIndex);
	}
}

void GenerateMatchingList(std::vector<std::vector<Point> > UsefulContoursL, std::vector<std::vector<Point> > UsefulContoursR, std::vector<Match> &Matcher) {
	unsigned int MatchCounter = 0;
	if ((UsefulContoursL.size()&& UsefulContoursR.size() )!= 0) {
		unsigned int i = 0;
		unsigned int j = 0;
		while (i < (UsefulContoursL).size()) {
			j = 0;
			while (j < (UsefulContoursR).size()) {
				double DMatchValue = 1;
				double SizeMatch = 0;
				DMatchValue = matchShapes(UsefulContoursL[i], UsefulContoursR[j], 1, 0.0);
				SizeMatch = ((contourArea(UsefulContoursL[i]) - contourArea(UsefulContoursR[j])) / (((contourArea(UsefulContoursL[i]) + (contourArea(UsefulContoursR[j]))) / 2)));
				if (SizeMatch < 0) {
					SizeMatch = -SizeMatch;
				}
				DMatchValue += SizeMatch;

				if (DMatchValue < 0.75) {//Is it at least a partial match?
					Matcher.push_back({ i,j,DMatchValue });
					MatchCounter++;
				}
				j++;
			}
			i++;
		}
	}
}

void ResolveMatchList(std::vector<Match> Matcher, std::vector<Match> &TentativeMatch) {
	unsigned int MatchCounter = 0;
	unsigned int TentativeMatchCounter = 0;
	bool AnyConflict = true;
	std::vector<Match> DeassignedMatch;
	TentativeMatch.clear();
	DeassignedMatch.clear();
	while (AnyConflict) {
		AnyConflict = false;
		if ((Matcher).size() != 0) {
			unsigned int i = 0;
			unsigned int j = 0;
			while (MatchCounter < (Matcher).size()) {
				//go through each match
				bool Conflict = false;
				i = 0;
				if ((TentativeMatch).size() != 0) {//check to see if any tentative matches have been made so far
					while (i < TentativeMatch.size()) { //Go through each previous result to see if there are any conflicts
						if ((TentativeMatch[i].LeftIndex == Matcher[MatchCounter].LeftIndex) || (TentativeMatch[i].RightIndex == Matcher[MatchCounter].RightIndex)) {//Do they use the same left or right index's?
							if (TentativeMatch[i].MatchValue > Matcher[MatchCounter].MatchValue) { //Is its match value lower?
																								   //Conflict detected
								DeassignedMatch.push_back(TentativeMatch[i]);//ReassignOldMatch
								j++;
								//
								TentativeMatch[i] = Matcher[MatchCounter];
								Conflict = true;
								AnyConflict = true;
							}
						}
						i++;
					}
					if (Conflict == false) {
						TentativeMatch.push_back(Matcher[MatchCounter]);
						TentativeMatchCounter++;
					}
				}
				else {//No tentative matches have been made so far
					TentativeMatch.push_back(Matcher[MatchCounter]);
					TentativeMatchCounter++;
				}
				MatchCounter++;
			}
		}
		Matcher.clear();
	}
}

void IDMatcher(std::vector <Match> InterframeMatchIndexes, std::vector <Match> OldInterframeMatchIndexes, std::vector <Point3i> &InterframeMatchIndexesComplete) {
	unsigned int i = 0;
	unsigned int j = 0;
	InterframeMatchIndexesComplete.clear();
	while (i<InterframeMatchIndexes.size()!=0) {
		j = 0;
		if (OldInterframeMatchIndexes.size()!=0) {
			while (j < OldInterframeMatchIndexes.size()!=0) {
				if ((InterframeMatchIndexes[i].RightIndex) == (OldInterframeMatchIndexes[j].LeftIndex)) {
					InterframeMatchIndexesComplete.push_back((Point3i)(InterframeMatchIndexes[i], OldInterframeMatchIndexes[j].RightIndex));
				}
				j++;
			}
		}
		i++;
	}
}

int CameraThread(bool CameraSide, CalibrationDataParameters CalibrationData, ColourSearchParameters SliderValue,
	std::vector<std::vector<Point>>* ImportedUsefulContoursOtherCamera,
	std::vector<std::vector<Point>> &ExportedUsefulContoursThisCamera,
	CenterPointDataParameters &ExportedCenterPointData,
	CenterPointDataParameters* ImportedCenterPointData, std::vector<double> &ExportedDist) {
	Mat SrcImgThisCamera;
	Mat CalibratedImgThisCamera;
	Mat HSVImageThisCamera;
	std::vector <Match> Matcher;
	std::vector <Match> TentativeMatch;
	std::vector <Match> DeassignedMatch;

	Mat ABSDiffSearchThesholdImageThisCamera;
	Mat ABSDiffPrevThisCamera;
	Mat ColourSearchThesholdImageThisCamera;
	Mat GrayForCannyThisCamera;

	Mat ABSDiffSearchDialatedThesholdImageThisCamera;
	Mat CannySearchThesholdImageThisCamera;
	Mat ThesholdImageThisCamera;
	Mat ColourSearchDialatedThesholdImageThisCamera;

	std::vector<std::vector<Point> > ContoursThisCamera;
	std::vector<std::vector<Point> > UsefulContoursThisCamera;
	std::vector<std::vector<Point> >PrevUsefulContoursThisCamera;

	std::vector<Vec4i> hierarchyThisCamera;
	std::vector<Match> PrevMatcherThisCamera;// (Left index,Right index , Match value)

	std::vector<Match> PrevTentativeMatchThisCamera;// (Left index,Right index , Match value)
	std::vector<std::vector<Point> > contoursInterFrameMatchOldThisCamera, contoursInterFrameMatchOlderThisCamera;

	std::vector<Match> InterFrameMatchOldThisCamera;
	std::vector<Match> TentativeInterFrameMatchOldThisCamera;
	
	Mat CannyPrevThisCamera;
	Mat CannyPrevThisCamera2;

	std::vector<Point2f> VectorCenter_pointThisCamera;
	std::vector<Point2f> OldVectorCenter_pointThisCamera;
	std::vector<Point2f> OlderVectorCenter_pointThisCamera;
	Point2f Center_pointThisCamera;
	std::chrono::steady_clock::time_point ImgTimeStampThisCamera;
	std::chrono::steady_clock::time_point OldImgTimeStampThisCamera;
	std::chrono::steady_clock::time_point OlderImgTimeStampThisCamera;

	VideoCapture capThisCamera;
	if (CameraSide == LeftCam) {
		capThisCamera.open(1);
	}else{
		capThisCamera.open(2);
	}
	auto CannySearchThread = async(CannySearch,CameraSide,&GrayForCannyThisCamera,ref(CannySearchThesholdImageThisCamera),ref(CannyPrevThisCamera), ref(CannyPrevThisCamera2));
	while (1) {
		Mat GrayThisCamera;
		vector<Mat> HSVChannels;

		Matcher.clear();
		TentativeMatch.clear();
		DeassignedMatch.clear();

		capThisCamera >> SrcImgThisCamera;

		OlderImgTimeStampThisCamera = OldImgTimeStampThisCamera;
		OldImgTimeStampThisCamera = ImgTimeStampThisCamera;
		ImgTimeStampThisCamera =std::chrono::high_resolution_clock::now();

		if (CameraSide == LeftCam) {
			LeftTimeStampMutex.lock();
		}else{
			RightTimeStampMutex.lock();
		}
		ExportedCenterPointData.ImgTimeStamp= ImgTimeStampThisCamera;
		ExportedCenterPointData.OldImgTimeStamp = OldImgTimeStampThisCamera;
		if (CameraSide == LeftCam) {
			LeftTimeStampMutex.unlock();
		}else{
			RightTimeStampMutex.unlock();
		}

		//Check image exists
		if (SrcImgThisCamera.empty()) {
			std::cout << "Error: Image cannot be loaded!" << endl;
			return -1;
		}
		//Apply calibrations
		if (CameraSide == LeftCam) {
			CalibrateLeftImage(SrcImgThisCamera, CalibrationData, CalibratedImgThisCamera);
		}else{
			CalibrateRightImage(SrcImgThisCamera, CalibrationData, CalibratedImgThisCamera);
		}
		cvtColor(CalibratedImgThisCamera, HSVImageThisCamera, CV_BGR2HSV);

		LightingCorrection(HSVImageThisCamera, HSVChannels, CalibratedImgThisCamera);

		cvtColor(CalibratedImgThisCamera, GrayThisCamera, CV_BGR2GRAY);

		//Start searching
		GrayForCannyThisCamera = GrayThisCamera;
		thread ABSDiffSearchThread(ABSDiffSearch, GrayThisCamera, ref(ABSDiffSearchThesholdImageThisCamera), ref(ABSDiffPrevThisCamera));
		thread ColourSearchThread(ColourSearch, HSVImageThisCamera, ref(ColourSearchThesholdImageThisCamera), SliderValue);

		ABSDiffSearchThread.join();
		ColourSearchThread.join();

		//L
		dilate(ColourSearchThesholdImageThisCamera, ColourSearchDialatedThesholdImageThisCamera, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
		subtract(ABSDiffSearchThesholdImageThisCamera, ColourSearchDialatedThesholdImageThisCamera, ThesholdImageThisCamera);
		addWeighted(ThesholdImageThisCamera, 1, ColourSearchThesholdImageThisCamera, 1, 0, ThesholdImageThisCamera);

		//Find contours
		findContours(ThesholdImageThisCamera, ContoursThisCamera, hierarchyThisCamera, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		FindUsefulContours(ContoursThisCamera, UsefulContoursThisCamera);
		//End finding contours

		//Extract the left useful contours from TentativeMatch
		if (!UsefulContoursThisCamera.empty()) {
			if (CameraSide == LeftCam) {
				LeftContourMutex.lock();
			}else{
				RightContourMutex.lock();
			}
			ExportedUsefulContoursThisCamera = UsefulContoursThisCamera;
			if (CameraSide == LeftCam) {
				LeftContourMutex.unlock();
			}else{
				RightContourMutex.unlock();
			}
			//Extract the left useful contours from PrevTentativeMatch
			if (PrevUsefulContoursThisCamera.empty()) {
				PrevUsefulContoursThisCamera = UsefulContoursThisCamera;
			}
			///////
			GenerateMatchingList(PrevUsefulContoursThisCamera, UsefulContoursThisCamera, PrevMatcherThisCamera);
			ResolveMatchList(PrevMatcherThisCamera, PrevTentativeMatchThisCamera);
			////////
			PrevUsefulContoursThisCamera = UsefulContoursThisCamera;

			PrevMatcherThisCamera.clear();
		}
		int InterframeMatchCounterL = 0;
		int InterframeMatchCounterR = 0;
		std::vector <Point3i> InterframeMatchIndexesCompleteThisCamera;

		if ((!UsefulContoursThisCamera.empty()) && (!PrevTentativeMatchThisCamera.empty())) {
			unsigned int i = 0;
			contoursInterFrameMatchOldThisCamera.clear();
			while (i < PrevTentativeMatchThisCamera.size()) {
				if (UsefulContoursThisCamera.size()>PrevTentativeMatchThisCamera[i].RightIndex) {//Check value is in range
					contoursInterFrameMatchOldThisCamera.push_back(UsefulContoursThisCamera[PrevTentativeMatchThisCamera[i].RightIndex]);
				}
				i++;
			}
			if (!contoursInterFrameMatchOldThisCamera.empty()) {
				GenerateMatchingList(contoursInterFrameMatchOldThisCamera, contoursInterFrameMatchOlderThisCamera, InterFrameMatchOldThisCamera);
				ResolveMatchList(InterFrameMatchOldThisCamera, TentativeInterFrameMatchOldThisCamera);

				IDMatcher(PrevTentativeMatchThisCamera, TentativeInterFrameMatchOldThisCamera, InterframeMatchIndexesCompleteThisCamera);
				contoursInterFrameMatchOlderThisCamera = contoursInterFrameMatchOldThisCamera;
			}
		}

		//Data trade
		std::vector<std::vector<Point>> UsefulContoursOtherCamera;
		if (CameraSide == RightCam) {
			LeftContourMutex.lock();
		}else{
			RightContourMutex.lock();
		}
		if (!(*ImportedUsefulContoursOtherCamera).empty()) {
			UsefulContoursOtherCamera = *ImportedUsefulContoursOtherCamera;
		}
		if (CameraSide == RightCam) {
			LeftContourMutex.unlock();
		}else{
			RightContourMutex.unlock();
		}

		PrevTentativeMatchThisCamera.clear();
		GenerateMatchingList(UsefulContoursThisCamera, UsefulContoursOtherCamera, Matcher);
		//Matcher
		ResolveMatchList(Matcher, TentativeMatch);
		
		//Find centre points
		if (!OldVectorCenter_pointThisCamera.empty()) {
			OlderVectorCenter_pointThisCamera = OldVectorCenter_pointThisCamera;
		}
		if (!VectorCenter_pointThisCamera.empty()) {
			OldVectorCenter_pointThisCamera = VectorCenter_pointThisCamera;
		}

		Mat drawingThisCamera = Mat::zeros((CalibratedImgThisCamera).size(), CV_8UC3);
		unsigned int MatchCounter = 0;
		std::vector<RotatedRect> minRectThisCamera((UsefulContoursThisCamera).size());
		unsigned int i=0;
		while (i < UsefulContoursThisCamera.size()) {
		//while (i < InterframeMatchIndexesCompleteL.size()) {
			Point2f rect_pointsThisCamera[4];
			Center_pointThisCamera = { 0,0 };
			Scalar color = Scalar(255, 255, 255);
			minRectThisCamera[TentativeMatch[MatchCounter].LeftIndex] = minAreaRect(Mat(UsefulContoursThisCamera[TentativeMatch[MatchCounter].LeftIndex]));
			drawContours(drawingThisCamera, UsefulContoursThisCamera, TentativeMatch[MatchCounter].LeftIndex, color, 1, 8, std::vector<Vec4i>(), 0, Point());
			//minRectL[TentativeMatch[MatchCounter].LeftIndex] = minAreaRect(Mat(UsefulContoursL[InterframeMatchIndexesCompleteL[i].x]));
			//drawContours(drawingL, UsefulContoursL, InterframeMatchIndexesCompleteL[i].x, color, 1, 8, std::vector<Vec4i>(), 0, Point());
			//Find the center point and draw rotated bounding box lines
			minRectThisCamera[TentativeMatch[MatchCounter].LeftIndex].points(rect_pointsThisCamera);
			for (int j = 0; j < 4; j++) {
				line(drawingThisCamera, rect_pointsThisCamera[j], rect_pointsThisCamera[(j + 1) % 4], color, 1, 8);
				Center_pointThisCamera += rect_pointsThisCamera[j];
			}
			Center_pointThisCamera /= 4;
			VectorCenter_pointThisCamera.push_back(Center_pointThisCamera);
			i++;
		}

		//Export data
		if (CameraSide == LeftCam) {
			LeftVectorCenterMutex.lock();
		}else{
			RightVectorCenterMutex.lock();
		}
		if (!InterframeMatchIndexesCompleteThisCamera.empty()) {
			((ExportedCenterPointData).InterframeMatchIndexesComplete) = InterframeMatchIndexesCompleteThisCamera;
		}
		if (!VectorCenter_pointThisCamera.empty()) {
			(ExportedCenterPointData.VectorCenter_point) = VectorCenter_pointThisCamera;
		}
		if (!OldVectorCenter_pointThisCamera.empty()) {
			(ExportedCenterPointData.OldVectorCenter_point) = OldVectorCenter_pointThisCamera;
		}
		if (!OlderVectorCenter_pointThisCamera.empty()) {
			(ExportedCenterPointData.OlderVectorCenter_point) = OlderVectorCenter_pointThisCamera;
		}
		if (CameraSide == LeftCam) {
			LeftVectorCenterMutex.unlock();
		}else {
			RightVectorCenterMutex.unlock();
		}
		//End export data

		//Import data
		std::vector<Point2f> VectorCenter_pointOtherCamera;
		std::vector<Point2f> OldVectorCenter_pointOtherCamera;
		std::vector<Point2f> OlderVectorCenter_pointOtherCamera;
		std::vector<Point3i> InterframeMatchIndexesCompleteOtherCamera;
		std::chrono::steady_clock::time_point ImgTimeStampOtherCamera;
		std::chrono::steady_clock::time_point OldImgTimeStampOtherCamera;
		std::chrono::steady_clock::time_point OlderImgTimeStampOtherCamera;

		if (CameraSide == RightCam) {
			LeftVectorCenterMutex.lock();
		}else{
			RightVectorCenterMutex.lock();
		}
		if(!((*ImportedCenterPointData).VectorCenter_point).empty()){
			VectorCenter_pointOtherCamera = (*ImportedCenterPointData).VectorCenter_point;
		}
		if (!((*ImportedCenterPointData).OldVectorCenter_point).empty()) {
			OldVectorCenter_pointOtherCamera = (*ImportedCenterPointData).OldVectorCenter_point;
		}
		if (!((*ImportedCenterPointData).OlderVectorCenter_point).empty()) {
			OlderVectorCenter_pointOtherCamera = (*ImportedCenterPointData).OlderVectorCenter_point;
		}
		if (!((*ImportedCenterPointData).InterframeMatchIndexesComplete).empty()) {
			InterframeMatchIndexesCompleteOtherCamera = (*ImportedCenterPointData).InterframeMatchIndexesComplete;
		}
		if (CameraSide == RightCam) {
			LeftVectorCenterMutex.unlock();
		}else{
			RightVectorCenterMutex.unlock();
		}
		//Importing time stamp info
		if (CameraSide == RightCam) {
			LeftTimeStampMutex.lock();
		}else{
			RightTimeStampMutex.lock();
		}
		ImgTimeStampOtherCamera = (*ImportedCenterPointData).ImgTimeStamp;
		OldImgTimeStampOtherCamera = (*ImportedCenterPointData).OldImgTimeStamp;
		OlderImgTimeStampOtherCamera = (*ImportedCenterPointData).OlderImgTimeStamp;
		if (CameraSide == RightCam) {
			LeftTimeStampMutex.unlock();
		}else{
			RightTimeStampMutex.unlock();
		}
		//End importing data

		//Calculate rights interpolated position
		std::vector<Point2f> InterpolatedVectorCenter_pointOtherCamera;
		i = 0;
		std::vector<double> dist;
		while (i < InterframeMatchIndexesCompleteOtherCamera.size()) {
			//Extract centerpoints from vectors
			Point2f InterpolationCalcCenter_pointR = VectorCenter_pointOtherCamera[InterframeMatchIndexesCompleteOtherCamera[i].x];
			Point2f InterpolationCalcOldCenter_pointR = OldVectorCenter_pointOtherCamera[InterframeMatchIndexesCompleteOtherCamera[i].y];
			Point2f InterpolationCalcOlderCenter_pointR = OlderVectorCenter_pointOtherCamera[InterframeMatchIndexesCompleteOtherCamera[i].z];

			//calc time diff
			steady_clock::duration InterpollationTimeDiffOne = OldImgTimeStampOtherCamera - OlderImgTimeStampOtherCamera;
			steady_clock::duration InterpollationTimeDiffTwo = ImgTimeStampOtherCamera - OldImgTimeStampOtherCamera;
			steady_clock::duration InterpollationTimeDiffThree = ImgTimeStampThisCamera - ImgTimeStampOtherCamera;

			//convert to float
			float InterpollationTimeDiffOne_nseconds = float(InterpollationTimeDiffOne.count()) * steady_clock::period::num / steady_clock::period::den;
			float InterpollationTimeDiffTwo_nseconds = float(InterpollationTimeDiffTwo.count()) * steady_clock::period::num / steady_clock::period::den;
			float InterpollationTimeDiffThree_nseconds = float(InterpollationTimeDiffThree.count()) * steady_clock::period::num / steady_clock::period::den;

			//calc interpolated centerpoint
			Point2f InterpollationPixelVelocityOne = (InterpolationCalcOldCenter_pointR - InterpolationCalcOlderCenter_pointR) / InterpollationTimeDiffOne_nseconds;
			Point2f InterpollationPixelVelocityTwo = (InterpolationCalcCenter_pointR - InterpolationCalcOldCenter_pointR) / InterpollationTimeDiffTwo_nseconds;
			Point2f InterpollationPixelAcceleration = (InterpollationPixelVelocityTwo - InterpollationPixelVelocityOne) / InterpollationTimeDiffTwo_nseconds;
			Point2f InterpollationPixelVelocityThree = InterpollationPixelVelocityTwo + (InterpollationPixelAcceleration * InterpollationTimeDiffThree_nseconds);
			Point2f InterpolatedCenterPointOtherCamera = (InterpollationPixelVelocityThree * InterpollationTimeDiffThree_nseconds) + InterpolationCalcCenter_pointR;
			//place centerpoint into vector
			InterpolatedVectorCenter_pointOtherCamera.push_back(InterpolatedCenterPointOtherCamera);

			//Calculate distance
			int disp;
			if (CameraSide == LeftCam) {
				disp = VectorCenter_pointThisCamera[i].x - InterpolatedVectorCenter_pointOtherCamera[i].x;
			}else {
				disp = InterpolatedVectorCenter_pointOtherCamera[i].x - VectorCenter_pointThisCamera[i].x;
			}
			dist.push_back(((201.6 * 4) / (disp*0.000043)) / 1000);
			i++;
		}
		if (CameraSide == LeftCam) {
			LeftDistanceMutex.lock();
		}else {
			RightDistanceMutex.lock();
		}
		ExportedDist = dist;
		if (CameraSide == LeftCam) {
			LeftDistanceMutex.unlock();
		}else{
			RightDistanceMutex.unlock();
		}
	}
}


int main() {
	bool DebugMode = false;
	Mat SrcImgL;
	Mat SrcImgR;
	Mat CalibratedImgL;
	Mat CalibratedImgR;

	CalibrationDataParameters CalibrationData;
	LoadCalibrationData(CalibrationData);

	namedWindow("SrcImgL", CV_WINDOW_AUTOSIZE);
	namedWindow("SrcImgR", CV_WINDOW_AUTOSIZE);

	namedWindow("ImgLCalibrated", CV_WINDOW_AUTOSIZE);
	namedWindow("ImgRCalibrated", CV_WINDOW_AUTOSIZE);


	//Set default slider positions
	ColourSearchParameters SliderValue;
	SliderValue.iLowHue = 0;
	SliderValue.iHighHue = 16;//16
	SliderValue.iLowHue2 = 158;//158
	SliderValue.iHighHue2 = 180;//180
	SliderValue.iLowSaturation = 141;//180
	SliderValue.iHighSaturation = 255;
	SliderValue.iLowValue = 38;//0
	SliderValue.iHighValue = 255;
	SliderValue.iMorphologicalOpening = 6;
	SliderValue.iMorphologicalClosing = 8;

	std::vector<std::vector<Point> > ContoursL;
	std::vector<std::vector<Point> > UsefulContoursL;

	std::vector<Vec4i> hierarchyL;
	std::vector<std::vector<Point> > ContoursR;
	std::vector<std::vector<Point> > UsefulContoursR;
	std::vector<Vec4i> hierarchyR;

	Point2f Center_pointL;
	Point2f Center_pointR;

	int iSliderValueDilation = 6;
	int iSliderValueCannyThres1 = 10;
	int iSliderValueCannyThres2 = 100;
	int iSliderValueCannyApt = 3;

	cv::Mat dxL, dyL;
	cv::Mat dxR, dyR;
	vector<Vec4i> linesL;
	vector<Vec4i> linesR;
	Mat ABSDiffPrevL;
	Mat ABSDiffPrevR;
	Mat CannyPrevL;
	Mat CannyPrevR;
	Mat CannyPrevL2;
	Mat CannyPrevR2;

	std::vector<std::vector<Point> > PrevUsefulContoursL;
	std::vector<std::vector<Point> > PrevUsefulContoursR;
	std::vector<std::vector<Point> >PrevUsefulContoursPL;
	std::vector<std::vector<Point> >PrevUsefulContoursPR;
	std::vector<std::vector<Point> >PrevUsefulContours;
	std::vector<Match>PrevMatcher;
	std::vector<Match>PrevMatcherP;

	std::vector<Match>PrevTentativeMatchP;

	std::vector<Match> PrevMatcherL;// (Left index,Right index , Match value)
	std::vector<Match> PrevMatcherR;// (Left index,Right index , Match value)

	std::vector<Match> PrevTentativeMatch;// (Left index,Right index , Match value)
	std::vector<Match> PrevTentativeMatchL;// (Left index,Right index , Match value)
	std::vector<Match> PrevTentativeMatchR;// (Left index,Right index , Match value)
	
	std::vector<Match> Matcher;// (Left index,Right index , Match value)
	std::vector<Match> TentativeMatch;// (Left index,Right index , Match value)
	std::vector<Match> DeassignedMatch;// (Left index,Right index , Match value)

	std::vector<std::vector<Point>> contoursInterFrameMatchOldL;
	std::vector<std::vector<Point>> contoursInterFrameMatchOlderL;
	std::vector<std::vector<Point>> contoursInterFrameMatchOldR;
	std::vector<std::vector<Point>> contoursInterFrameMatchOlderR;

	std::vector<Match>InterFrameMatchOldL;
	std::vector<Match>InterFrameMatchOldR;
	std::vector<Match>TentativeInterFrameMatchOldL;
	std::vector<Match>TentativeInterFrameMatchOldR;

	vector<Mat> HSVChannels;


	Mat HSVImageL;
	Mat HSVImageR;
	Mat ThesholdImageL;
	Mat ABSDiffSearchThesholdImageL;
	Mat CannySearchThesholdImageL;
	Mat ColourSearchThesholdImageL;

	Mat ABSDiffSearchDialatedThesholdImageL;
	Mat ColourSearchDialatedThesholdImageL;
	Mat ThesholdImageDialatedHueRangeL;

	Mat ThesholdImageR;
	Mat ABSDiffSearchThesholdImageR;
	Mat CannySearchThesholdImageR;
	Mat ColourSearchThesholdImageR;

	Mat ABSDiffSearchDialatedThesholdImageR;
	Mat ColourSearchDialatedThesholdImageR;
	Mat ThesholdImageDialatedHueRangeR;


	Mat GrayLForCanny, GrayRForCanny;
	bool CannySearchThreadStatusReady = true;
	unsigned int MainLoopCounter = 0;
	auto CannySearchThread = async(CannySearch, &GrayLForCanny, &GrayRForCanny, ref(CannySearchThesholdImageL), ref(CannySearchThesholdImageR), ref(CannyPrevL), ref(CannyPrevR), ref(CannyPrevL2), ref(CannyPrevR2));
	int LeftCamThread = async(CameraThread, LeftCam, CalibrationData, SliderValue,
		std::vector<std::vector<Point>> ImportedUsefulContoursOtherCamera,
		std::vector<std::vector<Point>> ExportedUsefulContoursThisCamera,
		CenterPointDataParameters ExportedCenterPointData,
		CenterPointDataParameters ImportedCenterPointData, std::vector<double> ExportedDist);
	while (1) {
		namedWindow("Sliders", CV_WINDOW_AUTOSIZE);
		namedWindow("Sliders2", CV_WINDOW_AUTOSIZE);
		createTrackbar("Hue:Low", "Sliders", &SliderValue.iLowHue, 180);
		createTrackbar("Hue:High", "Sliders", &SliderValue.iHighHue, 180);
		createTrackbar("Saturation:Low", "Sliders", &SliderValue.iLowSaturation, 255);
		createTrackbar("Saturation:High", "Sliders", &SliderValue.iHighSaturation, 255);
		createTrackbar("Value:Low", "Sliders", &SliderValue.iLowValue, 255);
		createTrackbar("Value:High", "Sliders", &SliderValue.iHighValue, 255);
		createTrackbar("MorphologicalOpening", "Sliders2", &SliderValue.iMorphologicalOpening, 30);
		createTrackbar("MorphologicalClosing", "Sliders2", &SliderValue.iMorphologicalClosing, 30);
		createTrackbar("Hue:Low2", "Sliders2", &SliderValue.iLowHue2, 180);
		createTrackbar("Hue:High2", "Sliders2", &SliderValue.iHighHue2, 180);

		createTrackbar("Dilation", "Sliders2", &iSliderValueDilation, 30);
		createTrackbar("Thres1", "Sliders2", &iSliderValueCannyThres1, 300);
		createTrackbar("Thres2", "Sliders2", &iSliderValueCannyThres2, 300);

		//Correct trackbars
		if (SliderValue.iMorphologicalOpening == 0) {
			SliderValue.iMorphologicalOpening = 1;
		}
		if (SliderValue.iMorphologicalClosing == 0) {
			SliderValue.iMorphologicalClosing = 1;
		}

		if (DebugMode == true) {
			imshow("SrcImgL", SrcImgL);
			imshow("SrcImgR", SrcImgR);

			imshow("ImgLCalibrated", CalibratedImgL);
			imshow("ImgRCalibrated", CalibratedImgR);

			imshow("ABSDiffSearchThesholdImageL", ABSDiffSearchThesholdImageL);
			imshow("ABSDiffSearchThesholdImageR", ABSDiffSearchThesholdImageR);

			imshow("ColourSearchThesholdImageL", ColourSearchThesholdImageL);
			imshow("ColourSearchThesholdImageR", ColourSearchThesholdImageR);
			if ((!CannySearchThesholdImageL.empty()) && (!CannySearchThesholdImageR.empty())) {
				imshow("CannySearchThesholdImageL", CannySearchThesholdImageL);
				imshow("CannySearchThesholdImageR", CannySearchThesholdImageR);
			}
		}else{
			destroyWindow("SrcImgL");
			destroyWindow("SrcImgR");

			destroyWindow("ImgLCalibrated");
			destroyWindow("ImgRCalibrated");

			destroyWindow("ABSDiffSearchThesholdImageL");
			destroyWindow("ABSDiffSearchThesholdImageR");

			destroyWindow("ColourSearchThesholdImageL");
			destroyWindow("ColourSearchThesholdImageR");

			destroyWindow("CannySearchThesholdImageL");
			destroyWindow("CannySearchThesholdImageR");
		}

		if (DebugMode == true) {
			imshow("FinalThresholdimgL", ThesholdImageL);
			imshow("FinalThresholdimgR", ThesholdImageR);
		}else {
			destroyWindow("FinalThresholdimgL");
			destroyWindow("FinalThresholdimgR");
		}



		unsigned int MatchCounter = 0;
		std::vector<RotatedRect> minRectL((UsefulContoursL).size());
		std::vector<RotatedRect> minRectR((UsefulContoursR).size());
		Mat drawingL = Mat::zeros((ThesholdImageL).size(), CV_8UC3);
		Mat drawingR = Mat::zeros((ThesholdImageR).size(), CV_8UC3);
		if (((UsefulContoursL).size() != 0)&&((UsefulContoursR).size() != 0)) {
			while (MatchCounter < (TentativeMatch).size()) {
				Scalar color = Scalar(0,(255 - (TentativeMatch[MatchCounter].MatchValue * 510)),(TentativeMatch[MatchCounter].MatchValue * 510));
				Point2f rect_pointsL[4];
				Point2f rect_pointsR[4];
				Center_pointL = { 0,0 };
				Center_pointR = { 0,0 };
				minRectL[TentativeMatch[MatchCounter].LeftIndex] = minAreaRect(Mat(UsefulContoursL[TentativeMatch[MatchCounter].LeftIndex]));
				drawContours(drawingL, UsefulContoursL, TentativeMatch[MatchCounter].LeftIndex, color, 1, 8, std::vector<Vec4i>(), 0, Point());
				//Find the center point and draw rotated bounding box lines
				minRectL[TentativeMatch[MatchCounter].LeftIndex].points(rect_pointsL);
				for (int j = 0; j < 4; j++) {
					line(drawingL, rect_pointsL[j], rect_pointsL[(j + 1) % 4], color, 1, 8);
					Center_pointL += rect_pointsL[j];
				}
				Center_pointL /= 4;
				cv::circle(drawingL, Center_pointL, 3, color, -3);	//Draws center point as a circle

				minRectR[TentativeMatch[MatchCounter].RightIndex] = minAreaRect(Mat(UsefulContoursR[TentativeMatch[MatchCounter].RightIndex]));
				drawContours(drawingR, UsefulContoursR, TentativeMatch[MatchCounter].RightIndex, color, 1, 8, std::vector<Vec4i>(), 0, Point());
				//Find the center point and draw rotated bounding box lines
				minRectR[TentativeMatch[MatchCounter].RightIndex].points(rect_pointsR);
				for (int j = 0; j < 4; j++) {
					line(drawingR, rect_pointsR[j], rect_pointsR[(j + 1) % 4], color, 1, 8);
					Center_pointR += rect_pointsR[j];
				}
				Center_pointR /= 4;
				cv::circle(drawingR, Center_pointR, 3, color, -3);	//Draws center point as a circle
				int disp = Center_pointL.x - Center_pointR.x;
				double dist = ((201.6 * 4) / (disp*0.000043)) / 1000;
				if (DebugMode == true) {
					printf("ID:%d\n", MatchCounter);
					printf("Disp:%d\n", disp);
					printf("Dist:%f cm\n", dist);
				}
				char textR[255];
				sprintf_s(textR, (const char)255, "Index: %d, Distance: %fcm",MatchCounter, dist);
				//sprintf_s(textR, (const char)255, "Index: %d", MatchCounter);
				putText(drawingR, textR, Center_pointR, FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200, 200, 250), 1, CV_AA); //Displays distance next object center point

				char textL[255];
				//sprintf_s(textL, (const char)255, "Index: %d", MatchCounter);
				sprintf_s(textL, (const char)255, "Index: %d, Distance: %fcm", MatchCounter, dist);
				putText(drawingL, textL, Center_pointL, FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200, 200, 250), 1, CV_AA); //Displays distance next object center point
				MatchCounter++;
			}
			cv::addWeighted(CalibratedImgL, 1, drawingL, 1, 0, CalibratedImgL);	//Overlays center point, contours, distance text and bounding box to "UndistotedImage"
			cv::addWeighted(CalibratedImgR, 1, drawingR, 1, 0, CalibratedImgR);	//Overlays center point, contours, distance text and bounding box to "UndistotedImage"
		}

		imshow("FinalImgL", CalibratedImgL);
		imshow("FinalImgR", CalibratedImgR);
		bool pause = false;
	
		switch (waitKey(10)) {
		case 27://esc ,exit program
			return 0;
		case 100://d, debug
			DebugMode = !DebugMode;
			break;
		case 112: //p, pause
			pause = !pause;
			if (pause == true) {
				cout << "Pause enabled,press 'p' again to resume" << endl;
				while (pause == true) {
					//stay in loop 
					switch (waitKey()) {
					case 112:
						pause = false;
						break;
					}
				}
			}
		}

		TentativeMatch.clear();
		UsefulContoursL.clear();
		waitKey(20);
	}
}




/*auto status = CannySearchThread.wait_for(chrono::milliseconds(0));
if (status == future_status::timeout) {
// still computing
CannySearchThreadStatusReady = false;
//printf("Still going\n");
}else if (status == future_status::ready) {
// finished computing
CannySearchThreadStatusReady = true;
printf("Done!!!!\n");
}else {
// There is still future_status::defered
printf("??\n\n");
}*/