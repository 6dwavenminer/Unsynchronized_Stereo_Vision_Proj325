#ifndef Main_HPP
#define Main_HPP
//Match[MatchCounter] =  { (double)i,(double)j,DMatchValue} ;
//std::vector<Point3d> TentativeMatch;// Left index,(Right index , Match value)
#include <string>

class Matcher{
     
    public:
    Matcher(MatchIndex, LeftIndex, RightIndex, MatchValue);       //constructor of the class
	unsigned int MatchIndex;
	unsigned int LeftIndex;
	unsigned int RightIndex;
	double MatchValue;
};

#endif /* Main_HPP */