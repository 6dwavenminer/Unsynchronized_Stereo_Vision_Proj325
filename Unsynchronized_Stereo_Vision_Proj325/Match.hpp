#ifndef Match_HPP
#define Match_HPP

class Match{
     
    public:
    Match(unsigned int LeftIndex, unsigned int  RightIndex,double MatchValue);       //constructor of the class
	unsigned int LeftIndex;
	unsigned int RightIndex;
	double MatchValue;

};

#endif /* Match_HPP */