#include "SearchAlgorithms.hpp""


/*
MorphilogicalFilter(Mat ThesholdImage)
Function takes input/output Mat ThresholdImage and attempt to remove noise by eroding and dilating the image
*/
void MorphilogicalFilter(Mat ThesholdImage) {
	erode(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(ThesholdImage, ThesholdImage, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
}


/*
ABSDiffSearch(Mat *Gray, Mat& ThesholdImage, Mat *ImportPrev,Mat& ExportPrev)
Function produces a threshold image Mat containing all movements by using a current grayscale image Mat and the previous grayscale image
Function also outputs a new grayscale image to be used next time
*/
void ABSDiffSearch(Mat *Gray, Mat& ThesholdImage, Mat *ImportPrev, Mat& ExportPrev) {

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

void ColourSearch(Mat *HSVImage, Mat& ThesholdImage, ColourSearchParameters *SliderValue) {
	Mat ThesholdImageInvertedHueRange;

	//Threshold Image
	inRange(*HSVImage, Scalar((*SliderValue).iLowHue, (*SliderValue).iLowSaturation, (*SliderValue).iLowValue), Scalar((*SliderValue).iHighHue, (*SliderValue).iHighSaturation, (*SliderValue).iHighValue), ThesholdImage);	//Hue,Saturation,Value
	inRange(*HSVImage, Scalar((*SliderValue).iLowHue2, (*SliderValue).iLowSaturation, (*SliderValue).iLowValue), Scalar((*SliderValue).iHighHue2, (*SliderValue).iHighSaturation, (*SliderValue).iHighValue), ThesholdImageInvertedHueRange);	//Hue,Saturation,Value
	addWeighted(ThesholdImage, 1, ThesholdImageInvertedHueRange, 1, 0, ThesholdImage);

	MorphilogicalFilter(ThesholdImage);
}

int CannySearch(bool CameraSide, Mat *ImportGrayThisCamera, std::vector<std::vector<Point>> *ImportCannyUsefulContoursOtherCamera, std::vector<Point2f> *ImportVectorCenter_pointOtherCamera, std::vector<std::vector<Point>>& ExportCannyUsefulContoursThisCamera, std::vector<Point2f>& ExportVectorCenter_pointThisCamera, Mat& ExportContourOverlay, Mat& ExportDebugCannyImg) {
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
	//std::vector<RotatedRect> minRectL;//((CannyUsefulContoursThisCamera).size());
	while (1) {
		VectorCenter_pointThisCamera.clear();
		VectorCenter_pointOtherCamera.clear();
		Matcher.clear();
		TentativeMatch.clear();
		DeassignedMatch.clear();

		while ((Pause == true) || (EnableCannySearch == false)) {}
		if (CloseProgram == true) {
			return 0;
		}

		if (CameraSide == LeftCam) {
			LeftCannyImportGrayMutex.lock();
		}
		else {
			RightCannyImportGrayMutex.lock();
		}
		if (!(*ImportGrayThisCamera).empty()) {
			GrayThisCamera = *ImportGrayThisCamera;
		}
		if (CameraSide == LeftCam) {
			LeftCannyImportGrayMutex.unlock();
		}
		else {
			RightCannyImportGrayMutex.unlock();
		}
		if (!(*ImportGrayThisCamera).empty()) {//////
			if (!PrevThisCamera.empty()) {
				//PrevThisCamera2 = PrevThisCamera;
				PrevThisCamera.copyTo(PrevThisCamera2);
			}
			if (!ThesholdImageThisCamera.empty()) {
				//PrevThisCamera = ThesholdImageThisCamera;
				ThesholdImageThisCamera.copyTo(PrevThisCamera);
			}
			Canny(GrayThisCamera, ThesholdImageThisCamera, 30, 300, 3);//8
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
			}
			else {
				LeftCannyTranferContoursMutex.unlock();
			}
			//End data trade
			if (CameraSide == LeftCam) {
				LeftCannyTranferContoursMutex.lock();
			}
			else {
				LeftCannyTranferContoursMutex.lock();
			}
			if (!(*ImportCannyUsefulContoursOtherCamera).empty()) {
				CannyUsefulContoursOtherCamera = *ImportCannyUsefulContoursOtherCamera;
			}
			if (CameraSide == LeftCam) {
				LeftCannyTranferContoursMutex.unlock();
			}
			else {
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
					}
					else {
						RightCannyTranferVectorCenterMutex.lock();
					}
					if (!VectorCenter_pointThisCamera.empty()) {
						ExportVectorCenter_pointThisCamera = VectorCenter_pointThisCamera;
					}
					if (CameraSide == RightCam) {
						LeftCannyTranferVectorCenterMutex.unlock();
					}
					else {
						RightCannyTranferVectorCenterMutex.unlock();
					}
					if (CameraSide == LeftCam) {
						LeftCannyTranferVectorCenterMutex.lock();
					}
					else {
						RightCannyTranferVectorCenterMutex.lock();
					}
					if (!(*ImportVectorCenter_pointOtherCamera).empty()) {
						VectorCenter_pointOtherCamera = *ImportVectorCenter_pointOtherCamera;
					}
					if (CameraSide == LeftCam) {
						LeftCannyTranferVectorCenterMutex.unlock();
					}
					else {
						RightCannyTranferVectorCenterMutex.unlock();
					}
					if (CameraSide == LeftCam) {
						if ((!VectorCenter_pointOtherCamera.empty()) && (VectorCenter_pointOtherCamera.size()>MatchCounter)) {
							disp = (int)(Center_pointThisCamera.x - VectorCenter_pointOtherCamera[MatchCounter].x);
						}
						else {
							disp = 0;
						}
					}
					else {
						if ((!VectorCenter_pointOtherCamera.empty()) && (VectorCenter_pointOtherCamera.size()>MatchCounter)) {
							disp = (int)(VectorCenter_pointOtherCamera[MatchCounter].x - Center_pointThisCamera.x);
						}
						else {
							disp = 0;
						}
					}
					dist = ((201.6 * 4) / (disp*0.000043)) / 1000;
					if (dist > 20) {
						sprintf_s(textThisCamera, (const char)255, "Index: %d, Distance: %.1fcm", MatchCounter, dist);
					}
					else {
						sprintf_s(textThisCamera, (const char)255, "Index: %d", MatchCounter);
					}
					putText(drawingThisCamera, textThisCamera, Center_pointThisCamera, FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(200, 200, 250), 1, CV_AA); //Displays distance next object center point
					if (!VectorCenter_pointOtherCamera.empty()) {
						//printf("Canny/C:%d,VectorCenter:%d\n", CameraSide, VectorCenter_pointOtherCamera[MatchCounter].x);
					}
					MatchCounter++;
					//minRectL.clear();
					VectorCenter_pointOtherCamera.clear();
				}
				VectorCenter_pointThisCamera.clear();
				//ExportContourOverlay = drawingThisCamera;
				if (CameraSide == LeftCam) {
					LeftCannyTranferContourOverlayMutex.lock();
				}
				else {
					RightCannyTranferContourOverlayMutex.lock();
				}
				drawingThisCamera.copyTo(ExportContourOverlay);
				if (CameraSide == LeftCam) {
					LeftCannyTranferContourOverlayMutex.unlock();
				}
				else {
					RightCannyTranferContourOverlayMutex.unlock();
				}
				minRectL.clear();
			}
		}
		this_thread::sleep_for(200ms);
	}
}
