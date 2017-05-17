/*

create by rosie
2017.03.30

*/


#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <string>
#include "iconRecog.h"	

using namespace std;
using namespace cv;


int main() {

	iconRecog icon;

	icon.HOGfeature2XML(); // 특징 추출
	
	icon.trainingBySVM(); // 트레이닝
	
	icon.testSVMTrainedData(); // 성능평가

	return 0;

}

