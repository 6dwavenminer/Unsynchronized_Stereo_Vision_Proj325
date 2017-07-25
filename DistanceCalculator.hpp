#ifndef DistanceCalculator_HPP
#define DistanceCalculator_HPP


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

void void CooridinatePositionCalculator(bool CameraSide, std::vector<double> dist, std::vector<Point2f> VectorCenter_pointThisCamera, vector<Point3d> &PoscmFromReferencePointVector);


#endif /* Match_HPP */