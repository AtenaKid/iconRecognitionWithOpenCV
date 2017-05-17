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

	// HOG descriptor parameter
	int WIN = 32;
	int BLOCK = 16;
	int STRIDE = 8;
	int CELL = 8;
	int BIN = 9;
	
	int trainPosDataNum = 55;
	int totalPosDataNum = 60; 
	
	int trainNegDataNum = 310;
	int totalNegDataNum = 318; //318
	
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
		"./icon_image/back/back/back",
		"./icon_image/close/close/close",
		"./icon_image/delete/delete/delete",
		"./icon_image/download/download/download",
		"./icon_image/edit/edit/edit",
		"./icon_image/home/home/home",
		"./icon_image/info/info/info",
		"./icon_image/love/love/love",
		"./icon_image/menu/menu/menu",
		"./icon_image/minus/minus/minus",
		"./icon_image/plus/plus/plus",
		"./icon_image/profile/profile/profile",
		"./icon_image/search/search/search",
		"./icon_image/settings/settings/settings",
		"./icon_image/share/share/share",
		"./icon_image/shopBag/shopBag/shopBag",
		"./icon_image/shopping/shopping/shopping",
		"./icon_image/negative/negative/negative"
	};


public:

	void HOGfeature2XML();
	void trainingBySVM();
	void testSVMTrainedData();
	Mat getHogDescriptorVisual(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);

};