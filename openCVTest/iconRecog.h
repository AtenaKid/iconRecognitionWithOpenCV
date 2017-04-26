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
	int trainPosDataNum = 46;
	int testPosDataNum = 5;
	int trainNegDataNum = 313;
	int testNegDataNum = 5;

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

	vector<string> trainFileName = {
		"./icon_image/train/back/back/back",
		"./icon_image/train/close/close/close",
		"./icon_image/train/delete/delete/delete",
		"./icon_image/train/download/download/download",
		"./icon_image/train/edit/edit/edit",
		"./icon_image/train/home/home/home",
		"./icon_image/train/info/info/info",
		"./icon_image/train/love/love/love",
		"./icon_image/train/menu/menu/menu",
		"./icon_image/train/minus/minus/minus",
		"./icon_image/train/plus/plus/plus",
		"./icon_image/train/profile/profile/profile",
		"./icon_image/train/search/search/search",
		"./icon_image/train/settings/settings/settings",
		"./icon_image/train/share/share/share",
		"./icon_image/train/shopBag/shopBag/shopBag",
		"./icon_image/train/shopping/shopping/shopping",
		"./icon_image/train/negative/negative/negative"
	};

	vector<string> evalFileName = {
		"./icon_image/eval/back/back/back",
		"./icon_image/eval/close/close/close",
		"./icon_image/eval/delete/delete/delete",
		"./icon_image/eval/download/download/download",
		"./icon_image/eval/edit/edit/edit",
		"./icon_image/eval/home/home/home",
		"./icon_image/eval/info/info/info",
		"./icon_image/eval/love/love/love",
		"./icon_image/eval/menu/menu/menu",
		"./icon_image/eval/minus/minus/minus",
		"./icon_image/eval/plus/plus/plus",
		"./icon_image/eval/profile/profile/profile",
		"./icon_image/eval/search/search/search",
		"./icon_image/eval/settings/settings/settings",
		"./icon_image/eval/share/share/share",
		"./icon_image/eval/shopBag/shopBag/shopBag",
		"./icon_image/eval/shopping/shopping/shopping",
		"./icon_image/eval/negative/negative/negative"
	};

public:

	void HOGfeature2XML();
	void trainingBySVM();
	void testSVMTrainedData();
};