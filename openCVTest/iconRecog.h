#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class iconRecog {

private:
	int classifyNum = 17;
	
	int trainPosDataNum = 30;
	int totalPosDataNum = 41; //41
	
	int trainNegDataNum = 50;
	int totalNegDataNum = 100; //318
	
	vector<string> hogFileName = {
		"PositiveBack.xml",
		"PositiveClose.xml",
		"PositiveDelete.xml",
		"PositiveDownload.xml",
		"PositiveEdit.xml",
		"PositiveHome.xml",
		"PositiveInfo.xml",
		"PositiveLove.xml",
		"PositiveMenu.xml",
		"PositiveMinus.xml",
		"PositivePlus.xml",
		"PositiveProfile.xml",
		"PositiveSearch.xml",
		"PositiveSettings.xml",
		"PositiveShare.xml",
		"PositiveShopBag.xml",
		"PositiveShopping.xml",
		"Negative.xml"
	};


	vector<string> iconClass = {
		"back",
		"close",
		"delete",
		"download",
		"edit",
		"home",
		"info",
		"love",
		"menu",
		"minus",
		"plus",
		"profile",
		"search",
		"settings",
		"share",
		"shopbag",
		"shopping",
		"negative"
	};

	vector<string> FileName = {
		"./icon_image/back/back",
		"./icon_image/close/close",
		"./icon_image/delete/delete",
		"./icon_image/download/download",
		"./icon_image/edit/edit",
		"./icon_image/home/home",
		"./icon_image/info/info",
		"./icon_image/love/love",
		"./icon_image/menu/menu",
		"./icon_image/minus/minus",
		"./icon_image/plus/plus",
		"./icon_image/profile/profile",
		"./icon_image/search/search",
		"./icon_image/settings/settings",
		"./icon_image/share/share",
		"./icon_image/shopBag/shopBag",
		"./icon_image/shopping/shopping",
		"./icon_image/negative/negative"
	};


public:

	void HOGfeature2XML();
	void trainingBySVM();
	void testSVMTrainedData();
};