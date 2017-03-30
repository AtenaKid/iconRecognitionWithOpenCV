#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>

using namespace std;

class iconRecog {

private:
	int classifyNum = 8;
	int trainDataNum = 35;
	int totalDataNum = 40;
	int backgroundNum = 27;

	vector<string> saveHogDesFileName = {
		"PositiveClose.xml",
		"PositiveBack.xml",
		"PositiveHome.xml",
		"PositiveMenu.xml",
		"PositiveProfile.xml",
		"PositiveSearch.xml",
		"PositiveSettings.xml",
		"PositiveShopping.xml",
		"Negative.xml"
	};


	vector<string> iconClass = {
		"close",
		"back",
		"home",
		"menu",
		"profile",
		"search",
		"settings",
		"shopping",
		"negative"
	};

	vector<string> FirstFileName = {
		"./images/close/close",
		"./images/back/back",
		"./images/home/home",
		"./images/menu/menu",
		"./images/profile/profile",
		"./images/search/search",
		"./images/settings/settings",
		"./images/shopping/shopping",
		"./images/negative/negative"
	};

public:
	const int SUM_MODE = 0;
	void addDataSet(int mode);
	void HOGfeature2XML();
	void trainingBySVM();
	void testSVMTrainedData();
	void testOneData(string filename, int iconIndex);

};