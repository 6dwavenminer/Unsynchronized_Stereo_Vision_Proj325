#include "DistanceCalculator.hpp"

//Global control variables
/*
*/
bool CoordinateDisplay = false;

double deg2rad(double deg) {
	return deg * PI / 180.0;
}
double rad2deg(double rad) {
	return rad * 180 / PI;
}

void MovingObjectDistanceCalculator(bool CameraSide, std::chrono::steady_clock::time_point ImgTimeStampThisCamera, std::vector<Point2f> VectorCenter_pointThisCamera,
	std::vector<Point2f> VectorCenter_pointOtherCamera,
	std::vector<Point2f> OldVectorCenter_pointOtherCamera,
	std::vector<Point2f> OlderVectorCenter_pointOtherCamera,
	std::vector<Point2f> InterpolatedVectorCenter_pointOtherCamera,
	std::vector<Point3i> InterframeMatchIndexesCompleteOtherCamera,
	std::chrono::steady_clock::time_point ImgTimeStampOtherCamera,
	std::chrono::steady_clock::time_point OldImgTimeStampOtherCamera,
	std::chrono::steady_clock::time_point OlderImgTimeStampOtherCamera,
	std::vector<double> &dist) {

	//Calculate interpolated position
	unsigned int i = 0;
	if ((!VectorCenter_pointOtherCamera.empty()) && (!OldVectorCenter_pointOtherCamera.empty()) && (!OlderVectorCenter_pointOtherCamera.empty())) {
		while (i < InterframeMatchIndexesCompleteOtherCamera.size()) {
			Point2f InterpolationCalcCenter_pointOtherCamera;
			Point2f InterpolationCalcOldCenter_pointOtherCamera;
			Point2f InterpolationCalcOlderCenter_pointOtherCamera;
			//Extract centerpoints from vectors
			if (VectorCenter_pointOtherCamera.size() >(unsigned)(InterframeMatchIndexesCompleteOtherCamera[i].x)) {
				InterpolationCalcCenter_pointOtherCamera = VectorCenter_pointOtherCamera[InterframeMatchIndexesCompleteOtherCamera[i].x];
			}
			else {
				InterpolationCalcCenter_pointOtherCamera = { 0,0 };
			}
			if (OldVectorCenter_pointOtherCamera.size() > (unsigned)(InterframeMatchIndexesCompleteOtherCamera[i].y)) {
				InterpolationCalcOldCenter_pointOtherCamera = OldVectorCenter_pointOtherCamera[InterframeMatchIndexesCompleteOtherCamera[i].y];
			}
			else {
				InterpolationCalcOldCenter_pointOtherCamera = { 0,0 };
			}
			if (OlderVectorCenter_pointOtherCamera.size() > (unsigned)(InterframeMatchIndexesCompleteOtherCamera[i].z)) {
				InterpolationCalcOlderCenter_pointOtherCamera = OlderVectorCenter_pointOtherCamera[InterframeMatchIndexesCompleteOtherCamera[i].z];
			}
			else {
				InterpolationCalcOlderCenter_pointOtherCamera = { 0,0 };
			}
			//calc time diff
			steady_clock::duration InterpollationTimeDiffOne = OldImgTimeStampOtherCamera - OlderImgTimeStampOtherCamera;
			steady_clock::duration InterpollationTimeDiffTwo = ImgTimeStampOtherCamera - OldImgTimeStampOtherCamera;
			steady_clock::duration InterpollationTimeDiffThree = ImgTimeStampThisCamera - ImgTimeStampOtherCamera;
			//convert to float
			float InterpollationTimeDiffOne_nseconds = float(InterpollationTimeDiffOne.count()) * steady_clock::period::num / steady_clock::period::den;
			float InterpollationTimeDiffTwo_nseconds = float(InterpollationTimeDiffTwo.count()) * steady_clock::period::num / steady_clock::period::den;
			float InterpollationTimeDiffThree_nseconds = float(InterpollationTimeDiffThree.count()) * steady_clock::period::num / steady_clock::period::den;
			//calc interpolated centerpoint
			Point2f InterpollationPixelVelocityOne = (InterpolationCalcOldCenter_pointOtherCamera - InterpolationCalcOlderCenter_pointOtherCamera) / InterpollationTimeDiffOne_nseconds;
			Point2f InterpollationPixelVelocityTwo = (InterpolationCalcCenter_pointOtherCamera - InterpolationCalcOldCenter_pointOtherCamera) / InterpollationTimeDiffTwo_nseconds;
			Point2f InterpollationPixelAcceleration = (InterpollationPixelVelocityTwo - InterpollationPixelVelocityOne) / InterpollationTimeDiffTwo_nseconds;
			Point2f InterpollationPixelVelocityThree = InterpollationPixelVelocityTwo + (InterpollationPixelAcceleration * InterpollationTimeDiffThree_nseconds);
			Point2f InterpolatedCenterPointOtherCamera = (InterpollationPixelVelocityThree * InterpollationTimeDiffThree_nseconds) + InterpolationCalcCenter_pointOtherCamera;
			//place centerpoint into vector
			InterpolatedVectorCenter_pointOtherCamera.push_back(InterpolatedCenterPointOtherCamera);
			//Calculate distance
			int dispx = 0;
			int dispy = 0;
			int disp = 0;
			if (!VectorCenter_pointThisCamera.empty()) {
				if ((VectorCenter_pointThisCamera.size()) > i) {
					if (CameraSide == LeftCam) {
						dispx = (int)(VectorCenter_pointThisCamera[i].x - InterpolatedVectorCenter_pointOtherCamera[i].x);
					}
					else {
						dispx = (int)(-VectorCenter_pointThisCamera[i].x + InterpolatedVectorCenter_pointOtherCamera[i].x);
					}
					dispy = (int)(VectorCenter_pointThisCamera[i].y - InterpolatedVectorCenter_pointOtherCamera[i].y);
					disp = (int)sqrt(pow(dispx, 2) + pow(dispy, 2));
				}
			}
			dist.push_back(pow(((10760 * pow(disp, -0.877)) / 3.0752), (1 / 0.7791)));
			i++;
		}
	}
}

void CooridinatePositionCalculator(bool CameraSide, std::vector<double> dist, std::vector<Point2f> VectorCenter_pointThisCamera, vector<Point3d> &PoscmFromReferencePointVector) {
	unsigned int i = 0;
	while ((dist.size() > i) && (VectorCenter_pointThisCamera.size() > i) && (CoordinateDisplay == true)) {
		double CameraViewAngleXY;
		double Camera2ObjectXYAngle;
		double CameraAngleDeviationFromReference;
		double Reference2ObjectXYAngle;
		double Camera2ObjectDistance;
		double CameraCentre2ObjectXYAngle;
		double XPoscmFromCamera;
		double XPoscmFromReferencePoint;
		double YPoscmFromReferencePoint;
		double CameraViewAngleZY;
		double ZPoscmFromReferencePoint;

		CameraViewAngleXY = ((double)VectorCenter_pointThisCamera[i].x / (double)XPixelDimensions) * (double)XYFOVangle;
		if (CameraSide == LeftCam) {
			CameraViewAngleXY = -(141.08*pow((dist[i]), -0.254) - CameraViewAngleXY + (55 - rad2deg(acos(10.08 / dist[i])))); //Apply calibrations
		}
		else {
			CameraViewAngleXY = (11.815*log(dist[i]) - 31.397 - CameraViewAngleXY + (125 - rad2deg(acos(10.08 / dist[i])))); //Apply calibrations
		}
		Camera2ObjectXYAngle = (double)125 - CameraViewAngleXY;
		CameraAngleDeviationFromReference = rad2deg(asin((sin(deg2rad(Camera2ObjectXYAngle)) / dist[i])*(double)(CameraDistcm / 2)));
		Reference2ObjectXYAngle = (double)180 - (Camera2ObjectXYAngle + CameraAngleDeviationFromReference);
		Camera2ObjectDistance = ((double)(CameraDistcm / 2) / sin(deg2rad(CameraAngleDeviationFromReference)))*sin(deg2rad(Reference2ObjectXYAngle));
		CameraCentre2ObjectXYAngle = (double)90 - Camera2ObjectXYAngle;
		XPoscmFromCamera = Camera2ObjectDistance*tan(deg2rad(CameraCentre2ObjectXYAngle));
		if (CameraSide == LeftCam) {
			XPoscmFromReferencePoint = XPoscmFromCamera - (double)(CameraDistcm / 2);
			XPoscmFromReferencePoint = (XPoscmFromReferencePoint + 24.401) / -1.6257;//Apply calibrations
		}
		else {
			XPoscmFromReferencePoint = XPoscmFromCamera + (double)(CameraDistcm / 2);
			XPoscmFromReferencePoint = (XPoscmFromReferencePoint - 34.3) / 1.6834;//Apply calibrations
		}
		YPoscmFromReferencePoint = sqrt(pow(dist[i], 2) - pow(XPoscmFromReferencePoint, 2));

		CameraViewAngleZY = (double)45 - (((double)VectorCenter_pointThisCamera[i].y / (double)YPixelDimensions) * (double)ZYFOVangle);
		ZPoscmFromReferencePoint = dist[i] * tan(deg2rad(CameraViewAngleZY));
		if (CameraSide == LeftCam) {
			ZPoscmFromReferencePoint = (ZPoscmFromReferencePoint - 0.6112) / 2.228;//Apply calibrations
		}
		else {
			ZPoscmFromReferencePoint = (ZPoscmFromReferencePoint - 6.3706) / 2.5771;//Apply calibrations
		}
		//Push points into vectors
		PoscmFromReferencePointVector.push_back({ XPoscmFromReferencePoint, YPoscmFromReferencePoint, ZPoscmFromReferencePoint });

		i++;
	}
}
