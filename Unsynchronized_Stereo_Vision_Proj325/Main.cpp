/*
Unsynchronized_Stereo_Vision_Proj325
Author: Oliver Thurgood
Version: 1.0.0
Created on: 05/02/2016
Last edited on: 12/05/2016
*/


#include <cstdio>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mutex>
#include <future>
#include <stdio.h> 
#include <math.h>
#include "Match.hpp"
#include "DistanceCalculator.hpp"

//Namespace list
using namespace cv;
using namespace std;
using namespace std::chrono;

//Mutex list
//////////////////////////////////////////////////////////
/*
Time stamp mutexs
-Allows for the safe transfer of time stamps between threads
*/

mutex LeftTimeStampMutex;
mutex RightTimeStampMutex;
//////////////////////////////////////////////////////////
/*
Contour mutexs
-Allows for the safe transfer of contours between threads
*/

mutex LeftContourMutex;
mutex RightContourMutex;
//////////////////////////////////////////////////////////
/*
Contour mutexs
-Allows for the safe transfer of object center points between threads
*/

mutex LeftVectorCenterMutex;
mutex RightVectorCenterMutex;
//////////////////////////////////////////////////////////
/*
Distance mutexs
-Allows for the safe transfer of object distances between threads
*/

mutex LeftDistanceMutex;
mutex RightDistanceMutex;
//////////////////////////////////////////////////////////
/*
Final image mutexs
-Allows for the safe transfer of the final left and right image matracies between threads
*/

mutex LeftFinalImgMutex;
mutex RightFinalImgMutex;
//////////////////////////////////////////////////////////
/*
Rolling barrier mutexs
-Used in main 'CameraThread' left and right threads
-Attempts to prevent threads lapping eachother without the need for full synchronisation
-Threads split into four blocks
-Each thread can only be in either the same block, or an adjacent block
-Achieved by locking block the in other thread which is not adjacent and unlocking the next adjacent block
*/

mutex LeftBlock1Mutex;
mutex LeftBlock2Mutex;
mutex LeftBlock3Mutex;
mutex LeftBlock4Mutex;
mutex RightBlock1Mutex;
mutex RightBlock2Mutex;
mutex RightBlock3Mutex;
mutex RightBlock4Mutex;
//////////////////////////////////////////////////////////
/*
Debug image mutexs
-Allows for the safe transfer of the debug threshold image matracies between threads
*/

mutex DebugColourThresImgLeftMutex;
mutex DebugColourThresImgRightMutex;
mutex DebugABSDiffThresImgLeftMutex;
mutex DebugABSDiffThresImgRightMutex;
mutex DebugCannyThresImgLeftMutex;
mutex DebugCannyThresImgRightMutex;			
//////////////////////////////////////////////////////////			
/*
Canny mutexs
-Allows for the safe transfer of information generated/required for canny edge detection between threads
*/

mutex LeftCannyImportGrayMutex;
mutex RightCannyImportGrayMutex;
mutex LeftCannyTranferContoursMutex;
mutex RightCannyTransferContoursMutex;
mutex LeftCannyTranferVectorCenterMutex;
mutex RightCannyTranferVectorCenterMutex;
mutex LeftCannyTranferContourOverlayMutex;
mutex RightCannyTranferContourOverlayMutex;
//////////////////////////////////////////////////////////

#define LeftCam true
#define RightCam false

#define XYFOVangle 70
#define ZYFOVangle 70
#define XPixelDimensions 640
#define YPixelDimensions 480
#define CameraDistcm 20.16 


//Global control variables

//Holds all threads in infinite loops while true
bool Pause = false;

//Enables the use of edge detection while true
bool EnableCannySearch = true;

//Enables the use of movement detection while true
bool EnableAbsdiffSearch = true;

//Enables the use of colour detection while true
bool EnableColourSearch = true;

/*
Enables debug mode while true
-Displays fps values for each camera on final displays
-Enables use of sliders to control the HSV threshold of the colour search
-Displays threshold images of all active search algorithms
*/
bool DebugMode = false;

/*
When true attempts to close program
-Force all threads to return a value of 0
-Due to OpenCV 3.0 bug, deconstructors on some variables cause the program to crash
*/
bool CloseProgram = false;

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

/*
CalibrationDataParameters
Stores matracies required for both left and right camera.

Mat intrinsicL, distCoeffsL, intrinsicR, distCoeffsR;
Mat RotationMat, TranslationMat, EssentailMat, FundamentalMat;
Mat RectificationTransformMatL, RectificationTransformMatR, ProjectionMatL, ProjectionMatR, Disparity2DepthMappingMat;
Mat map1x, map1y, map2x, map2y;
*/
struct CalibrationDataParameters {
	Mat intrinsicL, distCoeffsL, intrinsicR, distCoeffsR;
	Mat RotationMat, TranslationMat, EssentailMat, FundamentalMat;
	Mat RectificationTransformMatL, RectificationTransformMatR, ProjectionMatL, ProjectionMatR, Disparity2DepthMappingMat;
	Mat map1x, map1y, map2x, map2y;
};

/*
CenterPointDataParameters
Stores vectors and timestamps required for interpollating the position of objects.

std::vector<Point2f> VectorCenter_point;
std::vector<Point2f> OldVectorCenter_point;
std::vector<Point2f> OlderVectorCenter_point;
std::vector<Point3i> InterframeMatchIndexesComplete;
std::chrono::steady_clock::time_point ImgTimeStamp;
std::chrono::steady_clock::time_point OldImgTimeStamp;
std::chrono::steady_clock::time_point OlderImgTimeStamp;
*/
struct CenterPointDataParameters {
	std::vector<Point2f> VectorCenter_point;
	std::vector<Point2f> OldVectorCenter_point;
	std::vector<Point2f> OlderVectorCenter_point;
	std::vector<Point3i> InterframeMatchIndexesComplete;
	std::chrono::steady_clock::time_point ImgTimeStamp;
	std::chrono::steady_clock::time_point OldImgTimeStamp;
	std::chrono::steady_clock::time_point OlderImgTimeStamp;
};


/*
UserInput(void)
Function delays thread for 10ms to scan for any key presses to control the program using global bool variables:
p-Pause: Holds all threads in infinite loops while true

1-EnableCannySearch: Enables the use of edge detection while true
2-EnableAbsdiffSearch: Enables the use of movement detection while true
3-EnableColourSearch: Enables the use of colour detection while true

d-DebugMode: Enables debug mode while true
-Displays fps values for each camera on final displays
-Enables use of sliders to control the HSV threshold of the colour search
-Displays threshold images of all active search algorithms

esc-CloseProgram: When true attempts to close program
-Force all threads to return a value of 0
-Due to OpenCV 3.0 bug, deconstructors on some variables cause the program to crash
*/
void UserInput(void) {
	switch (waitKey(10)) {
	case 27://esc ,exit program
		CloseProgram = true;
		break;
	case 49://1, Toggle canny(edge detection) search
		if(EnableCannySearch==true){
			cout << "Canny search disabled, press '1' again to re-enable" << endl;
		}else {
			cout << "Canny search enabled, press '1' again to disable" << endl;
		}
		EnableCannySearch = !EnableCannySearch;
		break;
	case 50://2, Toggle absdiff(moving) search
		if (EnableAbsdiffSearch == true) {
			cout << "Absdiff(moving) search disabled, press '2' again to re-enable" << endl;
		}else {
			cout << "Absdiff(moving) search enabled, press '2' again to disable" << endl;
		}
		EnableAbsdiffSearch = !EnableAbsdiffSearch;
		break;
	case 51://3, Toggle colour(calibration) search
		if (EnableColourSearch == true) {
			cout << "Colour(calibration) search disabled, press '3' again to re-enable" << endl;
		}else {
			cout << "Colour(calibration) search enabled, press '3' again to disable" << endl;
		}
		EnableColourSearch = !EnableColourSearch;
		break;
	case 100://d, debug
		if (DebugMode == true) {
			cout << "Debug mode disabled, press 'd' again to re-enable" << endl;
		}else {
			cout << "Debug mode enabled, press 'd' again to disable" << endl;
		}
		DebugMode = !DebugMode;
		break;
	case 99://c, Coordinate display
		if (CoordinateDisplay == true) {
			cout << "Distance mode active, press 'c' to change to Coordinate mode" << endl;
		}else {
			cout << "Coordinate mode active, press 'c' to change to distance mode" << endl;
		}
		CoordinateDisplay = !CoordinateDisplay;
		break;
	case 112: //p, pause
		Pause = !Pause;
		if (Pause == true) {
			cout << "Pause enabled,press 'p' again to resume" << endl;
			while (Pause == true) {
				//stay in loop 
				switch (waitKey()) {
				case 112:
					cout << "Pause disabled, press 'p' again to pause" << endl;
					Pause = false;
					break;
				}
			}
		}
	}
}

/*
MorphilogicalFilter(Mat ThesholdImage)
Function takes input/output Mat ThresholdImage and attempt to remove noise by eroding and dilating the image
*/
void MorphilogicalFilter(Mat ThesholdImage){
	erode(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
}

/*
ABSDiffSearch(Mat *Gray, Mat& ThesholdImage, Mat *ImportPrev,Mat& ExportPrev)
Function produces a threshold image Mat containing all movements by using a current grayscale image Mat and the previous grayscale image
Function also outputs a new grayscale image to be used next time
*/
void ABSDiffSearch(Mat *Gray, Mat& ThesholdImage, Mat *ImportPrev,Mat& ExportPrev) {

	if ((*ImportPrev).empty()) {
		(*Gray).copyTo(*ImportPrev);
	}
	absdiff(*Gray, *ImportPrev, ThesholdImage);
	(*Gray).copyTo(ExportPrev);

	//Filter img
	threshold(ThesholdImage, ThesholdImage, 40, 255, THRESH_BINARY);
	MorphilogicalFilter(ThesholdImage);

	//joins back into parent thread
}

/*
void ColourSearch(Mat *HSVImage, Mat& ThesholdImage, ColourSearchParameters *SliderValue)
Produces a threshold image based on the current 'SliderValue'
*/
void ColourSearch(Mat *HSVImage, Mat& ThesholdImage, ColourSearchParameters *SliderValue) {
	Mat ThesholdImageInvertedHueRange;

	//Threshold Image
	inRange(*HSVImage, Scalar((*SliderValue).iLowHue, (*SliderValue).iLowSaturation, (*SliderValue).iLowValue), Scalar((*SliderValue).iHighHue, (*SliderValue).iHighSaturation, (*SliderValue).iHighValue), ThesholdImage);	//Hue,Saturation,Value
	inRange(*HSVImage, Scalar((*SliderValue).iLowHue2, (*SliderValue).iLowSaturation, (*SliderValue).iLowValue), Scalar((*SliderValue).iHighHue2, (*SliderValue).iHighSaturation, (*SliderValue).iHighValue), ThesholdImageInvertedHueRange);	//Hue,Saturation,Value
	addWeighted(ThesholdImage, 1, ThesholdImageInvertedHueRange, 1, 0, ThesholdImage);

	MorphilogicalFilter(ThesholdImage);
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

/*
void LightingCorrection(Mat HSVImage, vector<Mat> &HSVChannels,Mat &CalibratedImg)
Attempts to  correct image for current lighting conditions using 'equalizeHist'
*/
void LightingCorrection(Mat HSVImage, vector<Mat> &HSVChannels,Mat &CalibratedImg) {
	//vector<Mat> HSVChannels; cant be declared inside this function due to an OpenCV bug which fails to properally deconstruct it.
	split(HSVImage, HSVChannels);					//Split HSV into its constituant channels
	equalizeHist(HSVChannels[2], HSVChannels[2]);	//HSV value correction using histogram equalization
	merge(HSVChannels, HSVImage);					//Merge the value back into normal HSV format
	cvtColor(HSVImage, CalibratedImg, CV_HSV2BGR);	//Convert the HSV image back to BGR format
}

/*
void FindUsefulContours(std::vector<std::vector<Point> > &Contours,std::vector<std::vector<Point> > &UsefulContours)
Simplifies contours using approxPolyDP
Removes any objects that are below 30 pixels in size
*/
void FindUsefulContours(std::vector<std::vector<Point> > &Contours,std::vector<std::vector<Point> > &UsefulContours) {
	UsefulContours = Contours;
	if (Contours.size() != 0) {
		int iIndex = 0;
		int iUsefulIndex = 0;
		while ((unsigned)iIndex < Contours.size()) {
			if (contourArea(Contours[iIndex]) > 30) {
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

/*
void GenerateMatchingList(std::vector<std::vector<Point> > UsefulContoursL, std::vector<std::vector<Point> > UsefulContoursR, std::vector<Match> &Matcher)
Compares all contours of one vector with contours of the other assigning them a double value
Compares shape and size
Perfect match=0
Larger values indicate less liekly match
*/
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
				SizeMatch = abs((contourArea(UsefulContoursL[i]) - contourArea(UsefulContoursR[j])) / (((contourArea(UsefulContoursL[i]) + (contourArea(UsefulContoursR[j]))) / 2)));
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

/*
void ResolveMatchList(std::vector<Match> Matcher, std::vector<Match> &TentativeMatch)
Attempts to resolve the input match list using the Gale Shapely algorithm
*/
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

/*
void IDMatcher(std::vector <Match> InterframeMatchIndexes, std::vector <Match> OldInterframeMatchIndexes, std::vector <Point3i> &InterframeMatchIndexesComplete)
Takes two match vectors and attempts to match thier ID's to from a complete three stage match list
*/
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

/*
int CannySearch(bool CameraSide, Mat *ImportGrayThisCamera, std::vector<std::vector<Point>> *ImportCannyUsefulContoursOtherCamera,
std::vector<Point2f> *ImportVectorCenter_pointOtherCamera, std::vector<std::vector<Point>>& ExportCannyUsefulContoursThisCamera,
std::vector<Point2f>& ExportVectorCenter_pointThisCamera, Mat& ExportContourOverlay,Mat& ExportDebugCannyImg)

Once called continuously takes grayscale image and performs edge detection
Matches and draws objects and ouputs a contour overlay with any distance outputs
Sleeps for 200ms aat end of loop
*/
int CannySearch(bool CameraSide, Mat *ImportGrayThisCamera, std::vector<std::vector<Point>> *ImportCannyUsefulContoursOtherCamera,
	std::vector<Point2f> *ImportVectorCenter_pointOtherCamera, std::vector<std::vector<Point>>& ExportCannyUsefulContoursThisCamera,
	std::vector<Point2f>& ExportVectorCenter_pointThisCamera, Mat& ExportContourOverlay,Mat& ExportDebugCannyImg) {
	Mat GrayThisCamera;
	Mat PrevThisCamera;
	Mat PrevThisCamera2;
	Mat ThesholdImageThisCamera;
	std::vector <Match> Matcher;
	std::vector <Match> TentativeMatch;
	std::vector <Match> DeassignedMatch;
	std::vector<Vec4i> CannyhierarchyThisCamera;
	std::vector<std::vector<Point>> CannyContoursThisCamera;
	std::vector<std::vector<Point>> CannyUsefulContoursOtherCamera;
	std::vector<std::vector<Point>> CannyUsefulContoursThisCamera;

	Point2f Center_pointThisCamera;
	std::vector<Point2f> VectorCenter_pointThisCamera;
	std::vector<Point2f> VectorCenter_pointOtherCamera;

	Point2f rect_pointsL[4];
	Point2f rect_pointsR[4];
	Scalar color;
	double dist;
	int disp;
	char textThisCamera[255];
	while (1) {
		VectorCenter_pointThisCamera.clear();
		VectorCenter_pointOtherCamera.clear();
		Matcher.clear();
		TentativeMatch.clear();
		DeassignedMatch.clear();

		while ((Pause == true)||(EnableCannySearch==false)) {}
		if (CloseProgram == true) {
			return 0;
		}

		if (CameraSide == LeftCam) {
			LeftCannyImportGrayMutex.lock();
		}else {
			RightCannyImportGrayMutex.lock();
		}
		if (!(*ImportGrayThisCamera).empty()) {
			GrayThisCamera = *ImportGrayThisCamera;
		}
		if (CameraSide == LeftCam) {
			LeftCannyImportGrayMutex.unlock();
		}else {
			RightCannyImportGrayMutex.unlock();
		}
		if (!(*ImportGrayThisCamera).empty()) {
			if (!PrevThisCamera.empty()) {
				PrevThisCamera.copyTo(PrevThisCamera2);
			}
			if (!ThesholdImageThisCamera.empty()) {
				ThesholdImageThisCamera.copyTo(PrevThisCamera);
			}
			blur(GrayThisCamera, GrayThisCamera, Size(3, 3));
			Canny(GrayThisCamera, ThesholdImageThisCamera, 30, 300, 3);
			if (!PrevThisCamera.empty()) {
				if (!PrevThisCamera2.empty()) {
					cv::addWeighted(ThesholdImageThisCamera, 1, PrevThisCamera2, 1, 0, ThesholdImageThisCamera);
					cv::addWeighted(ThesholdImageThisCamera, 1, PrevThisCamera, 1, 0, ThesholdImageThisCamera);
				}
			}
			dilate(ThesholdImageThisCamera, ThesholdImageThisCamera, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));

			bitwise_not(ThesholdImageThisCamera, ThesholdImageThisCamera);

			threshold(ThesholdImageThisCamera, ThesholdImageThisCamera, 20, 255, THRESH_BINARY);

			Mat CannySearchDebugImg;
			if (DebugMode == true) {
				ThesholdImageThisCamera.copyTo(ExportDebugCannyImg);
			}

			//Find contours
			findContours(ThesholdImageThisCamera, CannyContoursThisCamera, CannyhierarchyThisCamera, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
			FindUsefulContours(CannyContoursThisCamera, CannyUsefulContoursThisCamera);
			//End finding contours
			//Data trade
			if (CameraSide == RightCam) {
				LeftCannyTranferContoursMutex.lock();
			}
			else {
				LeftCannyTranferContoursMutex.lock();
			}
			if (!CannyUsefulContoursThisCamera.empty()) {
				ExportCannyUsefulContoursThisCamera = CannyUsefulContoursThisCamera;
			}
			if (CameraSide == RightCam) {
				LeftCannyTranferContoursMutex.unlock();
			}else {
				LeftCannyTranferContoursMutex.unlock();
			}
			//End data trade
			if (CameraSide == LeftCam) {
				LeftCannyTranferContoursMutex.lock();
			}else {
				LeftCannyTranferContoursMutex.lock();
			}
			if (!(*ImportCannyUsefulContoursOtherCamera).empty()) {
				CannyUsefulContoursOtherCamera = *ImportCannyUsefulContoursOtherCamera;
			}
			if (CameraSide == LeftCam) {
				LeftCannyTranferContoursMutex.unlock();
			}else {
				LeftCannyTranferContoursMutex.unlock();
			}
			GenerateMatchingList(CannyUsefulContoursThisCamera, CannyUsefulContoursOtherCamera, Matcher);
			//Matcher
			ResolveMatchList(Matcher, TentativeMatch);
			unsigned int MatchCounter = 0;
			std::vector<RotatedRect> minRectL((CannyUsefulContoursThisCamera).size());
			RotatedRect minRectArea;
			Mat drawingThisCamera = Mat::zeros((ThesholdImageThisCamera).size(), CV_8UC3);
			Mat MinRectInputArray;
			if ((!(CannyUsefulContoursThisCamera).empty()) && (!(CannyUsefulContoursOtherCamera).empty())) {
				while (MatchCounter < (TentativeMatch).size()) {
					color = Scalar(0, (255 - (TentativeMatch[MatchCounter].MatchValue * 510)), (TentativeMatch[MatchCounter].MatchValue * 510));
					Center_pointThisCamera = { 0,0 };
					if ((TentativeMatch[MatchCounter].LeftIndex) < (CannyUsefulContoursThisCamera.size())) {
						MinRectInputArray = (Mat)(CannyUsefulContoursThisCamera[TentativeMatch[MatchCounter].LeftIndex]);
						//}
						if (!MinRectInputArray.empty()) {
							minRectArea = minAreaRect(MinRectInputArray);
						}
						if ((TentativeMatch[MatchCounter].LeftIndex) < (minRectL.size())) {
							minRectL[TentativeMatch[MatchCounter].LeftIndex] = minRectArea;
						}
						if ((TentativeMatch[MatchCounter].LeftIndex) < CannyUsefulContoursThisCamera.size()) {
							drawContours(drawingThisCamera, CannyUsefulContoursThisCamera, TentativeMatch[MatchCounter].LeftIndex, color, 1, 8, std::vector<Vec4i>(), 0, Point());
						}
						//Find the center point and draw rotated bounding box lines
						if ((TentativeMatch[MatchCounter].LeftIndex) < (minRectL.size())) {
							minRectL[TentativeMatch[MatchCounter].LeftIndex].points(rect_pointsL);
						}
						for (int j = 0; j < 4; j++) {
							line(drawingThisCamera, rect_pointsL[j], rect_pointsL[(j + 1) % 4], color, 1, 8);
							Center_pointThisCamera += rect_pointsL[j];
						}
						Center_pointThisCamera /= 4;
						VectorCenter_pointThisCamera.push_back(Center_pointThisCamera);
						cv::circle(drawingThisCamera, Center_pointThisCamera, 3, color, -3);	//Draws center point as a circle
					}
					if (CameraSide == RightCam) {
						LeftCannyTranferVectorCenterMutex.lock();
					}else {
						RightCannyTranferVectorCenterMutex.lock();
					}
					if (!VectorCenter_pointThisCamera.empty()) {
						ExportVectorCenter_pointThisCamera = VectorCenter_pointThisCamera;
					}
					if (CameraSide == RightCam) {
						LeftCannyTranferVectorCenterMutex.unlock();
					}else {
						RightCannyTranferVectorCenterMutex.unlock();
					}
					if (CameraSide == LeftCam) {
						LeftCannyTranferVectorCenterMutex.lock();
					}else {
						RightCannyTranferVectorCenterMutex.lock();
					}
					if (!(*ImportVectorCenter_pointOtherCamera).empty()) {
						VectorCenter_pointOtherCamera = *ImportVectorCenter_pointOtherCamera;
					}
					if (CameraSide == LeftCam) {
						LeftCannyTranferVectorCenterMutex.unlock();
					}else {
						RightCannyTranferVectorCenterMutex.unlock();
					}
					if (CameraSide == LeftCam) {
						if ((!VectorCenter_pointOtherCamera.empty()) && (VectorCenter_pointOtherCamera.size()>MatchCounter)) {
							disp = (int)(Center_pointThisCamera.x - VectorCenter_pointOtherCamera[MatchCounter].x);
						}else{
							disp = 0;
						}
					}else{
						if ((!VectorCenter_pointOtherCamera.empty()) && (VectorCenter_pointOtherCamera.size()>MatchCounter)) {
							disp = (int)(VectorCenter_pointOtherCamera[MatchCounter].x - Center_pointThisCamera.x);
						}else{
							disp = 0;
						}
					}
					dist = ((201.6 * 4) / (disp*0.000043)) / 1000;
					if (dist > 20) {
						sprintf_s(textThisCamera, (const char)255, "Index: %d, Distance: %.1fcm", MatchCounter, dist);
					}else {
						sprintf_s(textThisCamera, (const char)255, "Index: %d", MatchCounter);
					}
					putText(drawingThisCamera, textThisCamera, Center_pointThisCamera, FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200, 200, 250), 1, CV_AA); //Displays distance next object center point
					MatchCounter++;
					VectorCenter_pointOtherCamera.clear();
				}
				VectorCenter_pointThisCamera.clear();
				if (CameraSide == LeftCam) {
					LeftCannyTranferContourOverlayMutex.lock();
				}else {
					RightCannyTranferContourOverlayMutex.lock();
				}
				drawingThisCamera.copyTo(ExportContourOverlay);
				if (CameraSide == LeftCam) {
					LeftCannyTranferContourOverlayMutex.unlock();
				}else {
					RightCannyTranferContourOverlayMutex.unlock();
				}
				minRectL.clear();
			}
		}
		this_thread::sleep_for(200ms);
	}
}

/*
int CameraThread(bool CameraSide, CalibrationDataParameters CalibrationData, ColourSearchParameters *SliderValue,
std::vector<std::vector<Point>> *ImportedUsefulContoursOtherCamera,
std::vector<std::vector<Point>>& ExportedUsefulContoursThisCamera,
CenterPointDataParameters& ExportedCenterPointData,
CenterPointDataParameters *ImportedCenterPointData, std::vector<double>& ExportedDist,
std::vector<std::vector<Point>> *ImportCannyUsefulContoursOtherCamera,std::vector<Point2f> *ImportVectorCenter_pointOtherCamera,
std::vector<std::vector<Point>>& ExportCannyUsefulContoursThisCamera, std::vector<Point2f>& ExportVectorCenter_pointThisCamera,
Mat& ExportCannyContourOverlayThisCamera,
VideoCapture *capThisCamera,Mat& ExportedCalibratedImgThisCamera, Mat& ExportedDebugColourThresImgThisCamera, Mat& ExportedDebugABSDiffThresImgThisCamera,Mat& ExportDebugCannyImg)

Calls search threads
Matches objects and calculate distance to object
Draws contours and distance
*/
int CameraThread(bool CameraSide, CalibrationDataParameters CalibrationData, ColourSearchParameters *SliderValue,
	std::vector<std::vector<Point>> *ImportedUsefulContoursOtherCamera,
	std::vector<std::vector<Point>>& ExportedUsefulContoursThisCamera,
	CenterPointDataParameters& ExportedCenterPointData,
	CenterPointDataParameters *ImportedCenterPointData, std::vector<double>& ExportedDist,
	std::vector<std::vector<Point>> *ImportCannyUsefulContoursOtherCamera,std::vector<Point2f> *ImportVectorCenter_pointOtherCamera,
	std::vector<std::vector<Point>>& ExportCannyUsefulContoursThisCamera, std::vector<Point2f>& ExportVectorCenter_pointThisCamera,
	Mat& ExportCannyContourOverlayThisCamera,
	VideoCapture *capThisCamera,Mat& ExportedCalibratedImgThisCamera, Mat& ExportedDebugColourThresImgThisCamera, Mat& ExportedDebugABSDiffThresImgThisCamera,Mat& ExportDebugCannyImg){
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
	std::vector<std::vector<Point> > PrevUsefulContoursThisCamera;

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


	Mat ImportDebugCannyImg;
	auto CannySearchThreadThisCamera = async(CannySearch,CameraSide, &GrayForCannyThisCamera, &*ImportCannyUsefulContoursOtherCamera, &*ImportVectorCenter_pointOtherCamera, ref(ExportCannyUsefulContoursThisCamera), ref(ExportVectorCenter_pointThisCamera), ref(ExportCannyContourOverlayThisCamera),ref(ImportDebugCannyImg));
	Mat GrayThisCamera;
	vector<Mat> HSVChannels;

	if (CameraSide == LeftCam) {
		RightBlock2Mutex.lock();
		RightBlock3Mutex.lock();
	}else {
		LeftBlock2Mutex.lock();
		LeftBlock3Mutex.lock();
	}

	Mat ABSDiffPrevThisCamera2;
	int Current = 0;
	int Old = 0;


	std::vector<Point2f> VectorCenter_pointOtherCamera;
	std::vector<Point2f> OldVectorCenter_pointOtherCamera;
	std::vector<Point2f> OlderVectorCenter_pointOtherCamera;
	std::vector<Point2f> InterpolatedVectorCenter_pointOtherCamera;
	std::vector<Point3i> InterframeMatchIndexesCompleteOtherCamera;
	std::chrono::steady_clock::time_point ImgTimeStampOtherCamera;
	std::chrono::steady_clock::time_point OldImgTimeStampOtherCamera;
	std::chrono::steady_clock::time_point OlderImgTimeStampOtherCamera;

	while (1) {
		while (Pause == true) {}
		if (CloseProgram == true) {
			return 0;
		}

		/*
		Start of Block1
		*/
		if (CameraSide == LeftCam) {
			RightBlock4Mutex.lock();
			RightBlock2Mutex.unlock();
		}else {
			LeftBlock4Mutex.lock();
			LeftBlock2Mutex.unlock();
		}
		//Start making sure that all relavent vectors and empty.

		//Contour cannot be cleared due to OpenCV 3.0.0 bug
		/*while (!ContoursThisCamera.empty()) {
			ContoursThisCamera.clear();
		printf("C:%d,Test1a2\n", CameraSide);
		while (!UsefulContoursThisCamera.empty()) {
			UsefulContoursThisCamera.clear();
		}
		*/
		while (!InterFrameMatchOldThisCamera.empty()) {
			InterFrameMatchOldThisCamera.clear();
		}
		while (!TentativeInterFrameMatchOldThisCamera.empty()) {
			TentativeInterFrameMatchOldThisCamera.clear();
		}
		while (!Matcher.empty()) {
			Matcher.clear();
		}
		while (!TentativeMatch.empty()) {
			TentativeMatch.clear();
		}
		while (!DeassignedMatch.empty()) {
			DeassignedMatch.clear();
		}
		while (!VectorCenter_pointThisCamera.empty()) {
			VectorCenter_pointThisCamera.clear();
		}
		while (!VectorCenter_pointOtherCamera.empty()) {
			VectorCenter_pointOtherCamera.clear();
		}
		while (!OldVectorCenter_pointOtherCamera.empty()) {
			OldVectorCenter_pointOtherCamera.clear();
		}
		while (!OlderVectorCenter_pointOtherCamera.empty()) {
			OlderVectorCenter_pointOtherCamera.clear();
		}
		while (!InterpolatedVectorCenter_pointOtherCamera.empty()) {
			InterpolatedVectorCenter_pointOtherCamera.clear();
		}

		//End vector cleanup


		*capThisCamera >> SrcImgThisCamera;
		OlderImgTimeStampThisCamera = OldImgTimeStampThisCamera;
		OldImgTimeStampThisCamera = ImgTimeStampThisCamera;
		ImgTimeStampThisCamera =std::chrono::high_resolution_clock::now();
		steady_clock::duration FPSCalcTimeDiff = ImgTimeStampThisCamera - OldImgTimeStampThisCamera;
		//convert to float
		float FPSCalcTimeDiff_nseconds = float(FPSCalcTimeDiff.count()) * steady_clock::period::num / steady_clock::period::den;

		Mat drawingThisCamera = Mat::zeros((CalibratedImgThisCamera).size(), CV_8UC3);
		if (DebugMode == true) {
			char textThisCamera[255];
			sprintf_s(textThisCamera, (const char)255, "FPS: %f", 1/FPSCalcTimeDiff_nseconds);
			putText(drawingThisCamera, textThisCamera, Point2i(10,20), FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(255, 255, 255), 1, CV_AA); //Displays distance next object center point
		}




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
		if (CameraSide == LeftCam) {
			LeftCannyImportGrayMutex.lock();
		}else {
			RightCannyImportGrayMutex.lock();
		}
		GrayForCannyThisCamera = GrayThisCamera;
		if (CameraSide == LeftCam) {
			LeftCannyImportGrayMutex.unlock();
		}else {
			RightCannyImportGrayMutex.unlock();
		}


		if (ABSDiffPrevThisCamera.empty()) {
			GrayThisCamera.copyTo(ABSDiffPrevThisCamera);
		}

		if (ABSDiffPrevThisCamera2.empty()) {
			ABSDiffPrevThisCamera2.copyTo(ABSDiffPrevThisCamera);
		}

		if (CameraSide == LeftCam) {
			thread ABSDiffSearchThreadL(ABSDiffSearch, &GrayThisCamera, ref(ABSDiffSearchThesholdImageThisCamera),&ABSDiffPrevThisCamera, ref(ABSDiffPrevThisCamera2));
			thread ColourSearchThreadL(ColourSearch, &HSVImageThisCamera, ref(ColourSearchThesholdImageThisCamera), &*SliderValue);

			ABSDiffSearchThreadL.join();
			ColourSearchThreadL.join();
		}else{
			thread ABSDiffSearchThreadR(ABSDiffSearch, &GrayThisCamera, ref(ABSDiffSearchThesholdImageThisCamera), &ABSDiffPrevThisCamera,ref(ABSDiffPrevThisCamera2));
			thread ColourSearchThreadR(ColourSearch, &HSVImageThisCamera, ref(ColourSearchThesholdImageThisCamera), &*SliderValue);

			ABSDiffSearchThreadR.join();
			ColourSearchThreadR.join();

		}
		ABSDiffPrevThisCamera2.copyTo(ABSDiffPrevThisCamera);

		/*
		Start of Block2
		*/
		if (CameraSide == LeftCam) {
			RightBlock1Mutex.lock();
			RightBlock3Mutex.unlock();
		}else {
			LeftBlock1Mutex.lock();
			LeftBlock3Mutex.unlock();
		}
			
		//L
		//if to check for ColourSearchThesholdImageThisCamera
		if (ColourSearchThesholdImageThisCamera.empty()) {
			ColourSearchThesholdImageThisCamera=cv::Mat(SrcImgThisCamera.size(), CV_32F);//create blank
		}
		if ((EnableAbsdiffSearch == true) && (EnableColourSearch == true)) {
			dilate(ColourSearchThesholdImageThisCamera, ColourSearchDialatedThesholdImageThisCamera, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
			subtract(ABSDiffSearchThesholdImageThisCamera, ColourSearchDialatedThesholdImageThisCamera, ThesholdImageThisCamera);
			addWeighted(ThesholdImageThisCamera, 1, ColourSearchThesholdImageThisCamera, 1, 0, ThesholdImageThisCamera);
		}else if ((EnableAbsdiffSearch == true) && (EnableColourSearch == false)) {
			ABSDiffSearchThesholdImageThisCamera.copyTo(ThesholdImageThisCamera);
		}else if ((EnableColourSearch == true) && (EnableAbsdiffSearch == false)) {
			ColourSearchThesholdImageThisCamera.copyTo(ThesholdImageThisCamera);
		}else {
			ThesholdImageThisCamera=Mat::zeros((CalibratedImgThisCamera).size(), CV_8UC1);;
		}

		
		if (DebugMode == true) {
			if (CameraSide == LeftCam) {
				DebugColourThresImgLeftMutex.lock();
			}else{
				DebugColourThresImgRightMutex.lock();
			}
			if ((!ColourSearchThesholdImageThisCamera.empty()) && (EnableColourSearch == true)) {
				ColourSearchThesholdImageThisCamera.copyTo(ExportedDebugColourThresImgThisCamera);
			}
			if (CameraSide == LeftCam) {
				DebugColourThresImgLeftMutex.unlock();
			}else {
				DebugColourThresImgRightMutex.unlock();
			}

			if (CameraSide == LeftCam) {
				DebugABSDiffThresImgLeftMutex.lock();
			}else{
				DebugABSDiffThresImgRightMutex.lock();
			}
			if ((!ABSDiffSearchThesholdImageThisCamera.empty()) && (EnableAbsdiffSearch == true)) {
				ABSDiffSearchThesholdImageThisCamera.copyTo(ExportedDebugABSDiffThresImgThisCamera);
			}
			if (CameraSide == LeftCam) {
				DebugABSDiffThresImgLeftMutex.unlock();
			}else {
				DebugABSDiffThresImgRightMutex.unlock();
			}

			if (CameraSide == LeftCam) {
				DebugCannyThresImgLeftMutex.lock();
			}else {
				DebugCannyThresImgRightMutex.lock();
			}
			if ((!ImportDebugCannyImg.empty()) && (EnableCannySearch == true)) {
				ImportDebugCannyImg.copyTo(ExportDebugCannyImg);
			}
			if (CameraSide == LeftCam) {
				DebugCannyThresImgLeftMutex.unlock();
			}else {
				DebugCannyThresImgRightMutex.unlock();
			}
		}




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
				ResolveMatchList(InterFrameMatchOldThisCamera, TentativeInterFrameMatchOldThisCamera);//VERy slow
				IDMatcher(PrevTentativeMatchThisCamera, TentativeInterFrameMatchOldThisCamera, InterframeMatchIndexesCompleteThisCamera);
				InterFrameMatchOldThisCamera.clear();
				TentativeInterFrameMatchOldThisCamera.clear();
				contoursInterFrameMatchOlderThisCamera.clear();
				contoursInterFrameMatchOlderThisCamera = contoursInterFrameMatchOldThisCamera;
				contoursInterFrameMatchOldThisCamera.clear();
			}
		}
		/*
		Start of Block3
		*/
		if (CameraSide == LeftCam) {
			RightBlock2Mutex.lock();
			RightBlock4Mutex.unlock();
		}else {
			LeftBlock2Mutex.lock();
			LeftBlock4Mutex.unlock();
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
		unsigned int MatchCounter = 0;
		std::vector<RotatedRect> minRectThisCamera((UsefulContoursThisCamera).size());
		unsigned int i=0;
		if (!TentativeMatch.empty()) {
			while (MatchCounter < TentativeMatch.size()) {
				Point2f rect_pointsThisCamera[4];
				Center_pointThisCamera = { 0,0 };

				Scalar color = Scalar(0, (255 - (TentativeMatch[MatchCounter].MatchValue * 255)), (TentativeMatch[MatchCounter].MatchValue * 255));//510
				minRectThisCamera[TentativeMatch[MatchCounter].LeftIndex] = minAreaRect(Mat(UsefulContoursThisCamera[TentativeMatch[MatchCounter].LeftIndex]));
				drawContours(drawingThisCamera, UsefulContoursThisCamera, TentativeMatch[MatchCounter].LeftIndex, color, 1, 8, std::vector<Vec4i>(), 0, Point());

				//Find the center point and draw rotated bounding box lines
				minRectThisCamera[TentativeMatch[MatchCounter].LeftIndex].points(rect_pointsThisCamera);
				for (int j = 0; j < 4; j++) {
					line(drawingThisCamera, rect_pointsThisCamera[j], rect_pointsThisCamera[(j + 1) % 4], color, 1, 8);
					Center_pointThisCamera += rect_pointsThisCamera[j];
				}
				Center_pointThisCamera /= 4;
				cv::circle(drawingThisCamera, Center_pointThisCamera, 3, color, -3);	//Draws center point as a circle
				VectorCenter_pointThisCamera.push_back(Center_pointThisCamera);
				MatchCounter++;
			}
		}


		if (!OldVectorCenter_pointThisCamera.empty()) {
			OlderVectorCenter_pointThisCamera = OldVectorCenter_pointThisCamera;
		}
		if (!VectorCenter_pointThisCamera.empty()) {
			OldVectorCenter_pointThisCamera = VectorCenter_pointThisCamera;
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
		if (!OldVectorCenter_pointThisCamera.empty()) {
			OlderVectorCenter_pointThisCamera = OldVectorCenter_pointThisCamera;
		}
		if (!VectorCenter_pointThisCamera.empty()) {
			OldVectorCenter_pointThisCamera = VectorCenter_pointThisCamera;
		}


		/*
		Start of Block4
		*/
		if (CameraSide == LeftCam) {
			RightBlock3Mutex.lock();
			RightBlock1Mutex.unlock();
		}else {
			LeftBlock3Mutex.lock();
			LeftBlock1Mutex.unlock();
		}


		//Import data

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

		std::vector<double> dist;
		MovingObjectDistanceCalculator(CameraSide,ImgTimeStampThisCamera,VectorCenter_pointThisCamera,
			VectorCenter_pointOtherCamera,OldVectorCenter_pointOtherCamera,OlderVectorCenter_pointOtherCamera,
			InterpolatedVectorCenter_pointOtherCamera,InterframeMatchIndexesCompleteOtherCamera,
			ImgTimeStampOtherCamera,OldImgTimeStampOtherCamera,OlderImgTimeStampOtherCamera,
			dist);

		vector<Point3d>PoscmFromReferencePointVector;
		PoscmFromReferencePointVector.clear();

		CooridinatePositionCalculator(CameraSide,dist,VectorCenter_pointThisCamera,PoscmFromReferencePointVector);

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
		i = 0;
		if (!VectorCenter_pointThisCamera.empty()) {
			while ((i < dist.size())&&(i<VectorCenter_pointThisCamera.size())) {
				char textThisCamera[255];
				if (CoordinateDisplay == false) {
					sprintf_s(textThisCamera, (const char)255, "Index: %d, Distance: %.1fcm", i, dist[i]);
				}else {
					sprintf_s(textThisCamera, (const char)255, "Index: %d, X:%.1f,Y:%.1f,Z:%.1f cm", i, PoscmFromReferencePointVector[i].x, PoscmFromReferencePointVector[i].y, PoscmFromReferencePointVector[i].z);
				}
				putText(drawingThisCamera, textThisCamera, VectorCenter_pointThisCamera[i], FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200, 200, 250), 1, CV_AA); //Displays distance next object center point
				i++;
			}
		}
		VectorCenter_pointThisCamera.clear();//no longer required
		if (!drawingThisCamera.empty()) {
			cv::addWeighted(CalibratedImgThisCamera, 1, drawingThisCamera, 1, 0, CalibratedImgThisCamera);	//Overlays center point, contours, distance text and bounding box to "UndistotedImage"	
		}

		//Export img
		if (CameraSide == LeftCam) {
			LeftFinalImgMutex.lock();
		}else {
			RightFinalImgMutex.lock();
		}
		if (!CalibratedImgThisCamera.empty()) {
			if (CameraSide == LeftCam) {
				LeftCannyTranferContourOverlayMutex.lock();
			}else {
				RightCannyTranferContourOverlayMutex.lock();
			}
			if (!ExportCannyContourOverlayThisCamera.empty()) {
				if (EnableCannySearch == true) {
					cv::addWeighted(CalibratedImgThisCamera, 1, ExportCannyContourOverlayThisCamera, 1, 0, CalibratedImgThisCamera);	//Overlays center point, contours, distance text and bounding box to "UndistotedImage"
				}
			}
			if (CameraSide == LeftCam) {
				LeftCannyTranferContourOverlayMutex.unlock();
			}else {
				RightCannyTranferContourOverlayMutex.unlock();
			}
			CalibratedImgThisCamera.copyTo(ExportedCalibratedImgThisCamera);

		}
		if (CameraSide == LeftCam) {
			LeftFinalImgMutex.unlock();
		}else {
			RightFinalImgMutex.unlock();
		}
	}
}

/*
main
-startup
-user interface
*/
int main() {
	Mat SrcImgL;
	Mat SrcImgR;
	Mat CalibratedImgL;
	Mat CalibratedImgR;
	CalibrationDataParameters CalibrationData;
	LoadCalibrationData(CalibrationData);

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

	std::vector<std::vector<Point> > ContoursL;
	std::vector<std::vector<Point> > UsefulContoursL;

	std::vector<std::vector<Point> > ContoursR;
	std::vector<std::vector<Point> > UsefulContoursR;

	Point2f Center_pointL;
	Point2f Center_pointR;

	int iSliderValueDilation = 6;
	int iSliderValueCannyThres1 = 10;
	int iSliderValueCannyThres2 = 100;
	int iSliderValueCannyApt = 3;

	std::vector<Match>InterFrameMatchOldL;
	std::vector<Match>InterFrameMatchOldR;
	std::vector<Match>TentativeInterFrameMatchOldL;
	std::vector<Match>TentativeInterFrameMatchOldR;

	Mat GrayLForCanny, GrayRForCanny;
	bool CannySearchThreadStatusReady = true;
	unsigned int MainLoopCounter = 0;
	std::vector<std::vector<Point>> CannyUsefulContoursL;
	std::vector<std::vector<Point>> CannyUsefulContoursR;
	std::vector<Point2f> CannyVectorCenter_pointL;
	std::vector<Point2f> CannyVectorCenter_pointR;
	CenterPointDataParameters CenterPointDataL;
	CenterPointDataParameters CenterPointDataR;
	std::vector<double> DistL;
	std::vector<double> DistR;
	Mat CannyContourOverlayL;
	Mat CannyContourOverlayR;

	Mat srcL;
	Mat srcR;

	VideoCapture capCameraL;
	VideoCapture capCameraR;

	Mat ImportedFinalImgLeft;
	Mat ImportedFinalImgRight;
	Mat FinalImgLeft;
	Mat FinalImgRight;

	capCameraL.open(1);
	capCameraR.open(2);
	//Set resolution
	//capCameraL.set(CV_CAP_PROP_FRAME_WIDTH, XPixelDimensions);
	//capCameraL.set(CV_CAP_PROP_FRAME_HEIGHT, YPixelDimensions);
	//capCameraR.set(CV_CAP_PROP_FRAME_WIDTH, XPixelDimensions);
	//capCameraR.set(CV_CAP_PROP_FRAME_HEIGHT, YPixelDimensions);

	capCameraL >> SrcImgL;
	capCameraR >> SrcImgR;
	//Check image exists
	if (SrcImgL.empty()) {
		std::cout << "Error: Image cannot be loaded!" << endl;
		return -1;
	}
	//Check image exists
	if (SrcImgR.empty()) {
		std::cout << "Error: Image cannot be loaded!" << endl;
		return -1;
	}
	Mat ImportedDebugColourThresImgLeft;
	Mat ImportedDebugABSDiffThresImgLeft;
	Mat ImportedDebugCannyImgLeft;

	Mat ImportedDebugColourThresImgRight;
	Mat ImportedDebugABSDiffThresImgRight;
	Mat ImportedDebugCannyImgRight;

	auto LeftCamThread = async(CameraThread, LeftCam, CalibrationData, &SliderValue,
		&UsefulContoursR, ref(UsefulContoursL),
		ref(CenterPointDataL),&CenterPointDataR, ref(DistL),
		&CannyUsefulContoursR, &CannyVectorCenter_pointR,
		ref(CannyUsefulContoursL), ref(CannyVectorCenter_pointL),
		ref(CannyContourOverlayL), &capCameraL,ref(ImportedFinalImgLeft),
		ref(ImportedDebugColourThresImgLeft),ref(ImportedDebugABSDiffThresImgLeft),ref(ImportedDebugCannyImgLeft));
	auto RightCamThread = async(CameraThread, RightCam, CalibrationData, &SliderValue,
		&UsefulContoursL, ref(UsefulContoursR),
		ref(CenterPointDataR), &CenterPointDataL, ref(DistR),
		&CannyUsefulContoursL, &CannyVectorCenter_pointL,
		ref(CannyUsefulContoursR), ref(CannyVectorCenter_pointR),
		ref(CannyContourOverlayR), &capCameraR, ref(ImportedFinalImgRight),
		ref(ImportedDebugColourThresImgRight), ref(ImportedDebugABSDiffThresImgRight), ref(ImportedDebugCannyImgRight));

	cout << "Program start-up complete" << endl;
	cout << "User inputs:" << endl;
	cout << "p: Pause/unpause" << endl;
	cout << "d: Enable/disable debug mode" << endl;
	cout << "c: Distance/Co-ordiante  mode" << endl;
	cout << "esc: Close program" << endl;
	cout << "1: Enable/disable canny search" << endl;
	cout << "2: Enable/disable ABSDiff search" << endl;
	cout << "3: Enable/disable colour search" << endl;

	while (1) {

		LeftFinalImgMutex.lock();
		if (!ImportedFinalImgLeft.empty()) {
			ImportedFinalImgLeft.copyTo(FinalImgLeft);
		}
		LeftFinalImgMutex.unlock();

		RightFinalImgMutex.lock();
		if (!ImportedFinalImgRight.empty()) {
			ImportedFinalImgRight.copyTo(FinalImgRight);
		}
		RightFinalImgMutex.unlock();


		if (!FinalImgLeft.empty()) {
			imshow("FinalL", FinalImgLeft);
		}
		if (!FinalImgRight.empty()) {
			imshow("FinalR", FinalImgRight);
		}

		Mat DebugColourThresImgLeft;
		Mat DebugABSDiffThresImgLeft;
		Mat DebugCannyImgLeft;
		Mat DebugColourThresImgRight;
		Mat DebugABSDiffThresImgRight;
		Mat DebugCannyImgRight;

		if (DebugMode == true) {
			namedWindow("Sliders", CV_WINDOW_AUTOSIZE);
			createTrackbar("Hue:Low", "Sliders", &SliderValue.iLowHue, 180);
			createTrackbar("Hue:High", "Sliders", &SliderValue.iHighHue, 180);
			createTrackbar("Saturation:Low", "Sliders", &SliderValue.iLowSaturation, 255);
			createTrackbar("Saturation:High", "Sliders", &SliderValue.iHighSaturation, 255);
			createTrackbar("Value:Low", "Sliders", &SliderValue.iLowValue, 255);
			createTrackbar("Value:High", "Sliders", &SliderValue.iHighValue, 255);
			createTrackbar("Hue:Low2", "Sliders", &SliderValue.iLowHue2, 180);
			createTrackbar("Hue:High2", "Sliders", &SliderValue.iHighHue2, 180);

			if (EnableColourSearch == true) {
				DebugColourThresImgLeftMutex.lock();
				if (!ImportedDebugColourThresImgLeft.empty()) {
					ImportedDebugColourThresImgLeft.copyTo(DebugColourThresImgLeft);
					imshow("Debug:Colour search threshold: Left", DebugColourThresImgLeft);
				}
				DebugColourThresImgLeftMutex.unlock();
				DebugColourThresImgRightMutex.lock();
				if (!ImportedDebugColourThresImgRight.empty()) {
					ImportedDebugColourThresImgRight.copyTo(DebugColourThresImgRight);
					imshow("Debug:Colour search threshold: Right", DebugColourThresImgRight);
				}
				DebugColourThresImgRightMutex.unlock();
			}else{
				destroyWindow("Debug:Colour search threshold: Left");
				destroyWindow("Debug:Colour search threshold: Right");
			}
			if (EnableAbsdiffSearch == true) {
				DebugABSDiffThresImgLeftMutex.lock();
				if (!ImportedDebugABSDiffThresImgLeft.empty()) {
					ImportedDebugABSDiffThresImgLeft.copyTo(DebugABSDiffThresImgLeft);
					imshow("Debug:ABSDiff search threshold: Left", DebugABSDiffThresImgLeft);
				}
				DebugABSDiffThresImgLeftMutex.unlock();
				DebugABSDiffThresImgRightMutex.lock();
				if (!ImportedDebugABSDiffThresImgRight.empty()) {
					ImportedDebugABSDiffThresImgRight.copyTo(DebugABSDiffThresImgRight);
					imshow("Debug:ABSDiff search threshold: Right", DebugABSDiffThresImgRight);
				}
				DebugABSDiffThresImgRightMutex.unlock();
			}else{
				destroyWindow("Debug:ABSDiff search threshold: Left");
				destroyWindow("Debug:ABSDiff search threshold: Right");
			}
			if (EnableCannySearch == true) {
				DebugCannyThresImgLeftMutex.lock();
				if (!ImportedDebugCannyImgLeft.empty()) {
					ImportedDebugCannyImgLeft.copyTo(DebugCannyImgLeft);
					imshow("Debug:Canny search threshold: Left", DebugCannyImgLeft);
				}
				DebugCannyThresImgLeftMutex.unlock();
				DebugCannyThresImgRightMutex.lock();
				if (!ImportedDebugCannyImgRight.empty()) {
					ImportedDebugCannyImgRight.copyTo(DebugCannyImgRight);
					imshow("Debug:Canny search threshold: Right", DebugCannyImgRight);
				}
				DebugCannyThresImgRightMutex.unlock();
			}else{
				destroyWindow("Debug:Canny search threshold: Left");
				destroyWindow("Debug:Canny search threshold: Right");
			}
		}else{
			destroyWindow("Sliders");

			destroyWindow("Debug:Colour search threshold: Left");
			destroyWindow("Debug:Colour search threshold: Right");

			destroyWindow("Debug:ABSDiff search threshold: Left");
			destroyWindow("Debug:ABSDiff search threshold: Right");

			destroyWindow("Debug:Canny search threshold: Left");
			destroyWindow("Debug:Canny search threshold: Right");
		}


		UserInput();
		if (CloseProgram == true) {
			return 0;
		}
	}
}
