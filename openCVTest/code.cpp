/*

create by rosie
2017.03.30

*/

#include <iostream>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>
#include "iconRecog.h"


using namespace std;
using namespace cv;

/*

아이콘과 배경 이미지를 합성한 새로운 데이터를 생성

*/

void iconRecog::addDataSet(int mode) {

	for (int i = 0; i < classifyNum + 1; ++i) {

		char FullFileName[100];
		for (int j = 0; j < trainDataNum; ++j) {

			sprintf_s(FullFileName, "%s%d.png", FirstFileName[i].c_str(), j);

			printf("%s loaded! \n", FullFileName);

			Mat img = imread(FullFileName);

			if (mode == SUM_MODE) {

				char backImageName[100];
				char NewFileName[100];
				for (int k = 0; k < backgroundNum; k++) {

					sprintf_s(backImageName, "./images/background/background%d.png", k);
					printf("%s loaded! \n", backImageName);
					Mat backimg = imread(backImageName);

					resize(img, img, Size(64, 64), 0, 0, CV_INTER_LANCZOS4);
					resize(backimg, backimg, Size(64, 64), 0, 0, CV_INTER_LANCZOS4);

					Mat img_gray, mask, mask_inv;
					cvtColor(img, img_gray, CV_RGB2GRAY);
					threshold(img_gray, mask, 200, 255, THRESH_BINARY);
					bitwise_not(mask, mask_inv);

					Mat kernal = Mat::ones(2, 2, CV_8U);
					dilate(mask_inv, mask_inv, kernal);

					Mat img_bg, img_fg;
					bitwise_and(backimg, backimg, img_bg, mask = mask);
					bitwise_and(img, img, img_fg, mask = mask_inv);

					Mat sumImage;
					add(img_bg, img_fg, sumImage);
					sprintf_s(NewFileName, "%s%d_M%d.png", FirstFileName[i].c_str(), j, k);
					imwrite(NewFileName, sumImage);
				}

			}
		}
	}
}

/*

OPEN CV HOG를 통해 각 아이콘의 edge 특징을 xml 파일로 추출

*/


void iconRecog::HOGfeature2XML() {

	char FullFileName[100];

	for (int i = 0; i < (classifyNum + 1); ++i) {

		vector< vector < float> > v_descriptorsValues;
		vector< vector < Point> > v_locations;

		for (int j = 0; j < trainDataNum; ++j) {

			sprintf_s(FullFileName, "%s%d.png", FirstFileName[i].c_str(), j);

			printf("%s loaded! \n", FullFileName);

			Mat img, img_gray;
			img = imread(FullFileName);

			// --------------이미지 전처리-------------------

			resize(img, img, Size(32, 32), 0, 0, CV_INTER_LANCZOS4);

			cvtColor(img, img_gray, CV_RGB2GRAY);

			// ------------- 특징 추출 -----------------------

			HOGDescriptor d(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4), 9);

			vector< float> descriptorsValues;
			vector< Point> locations;
			d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			v_descriptorsValues.push_back(descriptorsValues);
			v_locations.push_back(locations);

			char NewFileName[100];
			for (int k = 0; k < backgroundNum; k++) {

				sprintf_s(NewFileName, "%s%d_M%d.png", FirstFileName[i].c_str(), j, k);

				printf("%s loaded! \n", NewFileName);
				Mat img, img_gray;
				img = imread(NewFileName);
				resize(img, img, Size(32, 32), 0, 0, CV_INTER_LANCZOS4);
				cvtColor(img, img_gray, CV_RGB2GRAY);
				HOGDescriptor d(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4), 9);

				vector< float> descriptorsValues;
				vector< Point> locations;
				d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

				v_descriptorsValues.push_back(descriptorsValues);
				v_locations.push_back(locations);

			}

		}

		// -------------------- 특징 정보 xml 데이터로 저장 ----------------

		FileStorage hogXml(saveHogDesFileName[i], FileStorage::WRITE);

		int row = v_descriptorsValues.size(), col = v_descriptorsValues[0].size();

		Mat M(row, col, CV_32F);
		for (int i = 0; i < row; ++i)
			memcpy(&(M.data[col * i * sizeof(float)]), v_descriptorsValues[i].data(), col * sizeof(float));

		write(hogXml, "Descriptor_of_images", M);
		hogXml.release();

	}

}

/*

각 아이콘의 특징 정보를 하나의 MAT 데이터롤 통합하여, 라벨 MAT을 정의하여
SVM을 통해  트레이닝
모델을 xml 데이터로 추출

*/

void iconRecog::trainingBySVM() {

	vector<Mat> positiveMat;
	// ------------- 특징 정보가 담긴 xml 데이터 Mat으로 읽어 들임 ---------------
	printf("1. feature data load \n");
	printf("%s feature data load \n", saveHogDesFileName[classifyNum].c_str());
	FileStorage read_NegativeXml(saveHogDesFileName[classifyNum], FileStorage::READ);
	//Negative Mat
	Mat nMat;
	read_NegativeXml["Descriptor_of_images"] >> nMat;

	int nRow, nCol;
	nRow = nMat.rows; nCol = nMat.cols;
	printf("rows: %d , cols: %d \n", nRow, nCol);

	read_NegativeXml.release();

	for (int i = 0; i < (int)saveHogDesFileName.size() - 1; ++i) {

		printf("%s feature data load \n", saveHogDesFileName[i].c_str());

		FileStorage read_PositiveXml(saveHogDesFileName[i], FileStorage::READ);

		//Positive Mat
		Mat pMat;
		read_PositiveXml["Descriptor_of_images"] >> pMat;
		positiveMat.push_back(pMat);
		printf("rows: %d , cols: %d \n", positiveMat[i].rows, positiveMat[i].cols);

		read_PositiveXml.release();

	}

	// ---------------- 여러 클래스 데이터를 하나의 Mat으로 통합------------------
	printf("2. Make training data for SVM\n");
	// rows == # of training data, cols == # of descriptor
	Mat trainigData(positiveMat[0].rows * classifyNum + nRow, nCol, CV_32FC1);
	Mat labels(nRow*(classifyNum + 1), 1, CV_32S, Scalar(classifyNum));

	printf("label row: %d, cols: %d \n", labels.rows, labels.cols);

	int startData = 0;
	int startLabel = 0;
	// positive data set
	for (int i = 0; i < (int)positiveMat.size(); ++i) {
		memcpy(&(trainigData.data[startData]), positiveMat[i].data, sizeof(float) * positiveMat[i].cols * nRow);
		labels.rowRange(startLabel, startLabel + nRow) = Scalar(i);
		startLabel = startLabel + nRow;
		startData = startData + sizeof(float) * positiveMat[i].cols * nRow; // 다음이 가르킬 위치
	}
	// negative data set
	memcpy(&(trainigData.data[startData]), nMat.data, sizeof(float) * nMat.cols * nMat.rows);

	// ------------------- svm 모델 생성 및 트레이닝 후 모델을 xml 데이터로 저장 -------------------------------
	printf("3. Set SVM parameter \n");

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm->setType(ml::SVM::NU_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setGamma(1);
	svm->setCoef0(0);
	svm->setNu(0.2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));

	printf("4. Training \n");

	svm->train(trainigData, ml::ROW_SAMPLE, labels);

	printf("5. SVM XML svae \n");
	svm->save("trainedSVM.xml");

}

/*

모델을 테스트 데이터로 테스트하여 성능 평가

*/


void iconRecog::testSVMTrainedData() {

	int trueResult = 0, falseResult = 0;

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm = ml::SVM::load("trainedSVM.xml");

	char FullFileName[100];

	// -------------- 테스트 데이터를 읽어들이면서 결과 예측 -----------------------------

	for (int i = 0; i <= classifyNum; ++i) {

		for (int j = trainDataNum; j < totalDataNum; ++j) {

			sprintf_s(FullFileName, "%s%d.png", FirstFileName[i].c_str(), j);

			Mat img, img_gray;
			img = imread(FullFileName);

			resize(img, img, Size(32, 32), 0, 0, CV_INTER_LANCZOS4);

			cvtColor(img, img_gray, CV_RGB2GRAY);


			HOGDescriptor d(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4), 9);

			vector<float> descriptorsValues;
			vector<Point> locations;
			d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			// ----------Classification----------------

			int row = 1, col = descriptorsValues.size();

			Mat M(row, col, CV_32F);
			memcpy(&(M.data[0]), descriptorsValues.data(), col * sizeof(float));

			int result = svm->predict(M);

			if (i < classifyNum && result == i) {
				printf("%s --> %s [true] \n", FullFileName, iconClass[result].c_str());
				trueResult++;
			}
			else if (i < classifyNum && result != i) {
				printf("%s --> %s [false] \n", FullFileName, iconClass[result].c_str());
				falseResult++;
			}
			else if (i == classifyNum && result >= i) {
				printf("%s --> %s [true] \n", FullFileName, iconClass[result].c_str());
				trueResult++;
			}
			else if (i == classifyNum && result < i) {
				printf("%s --> %s [false] \n", FullFileName, iconClass[result].c_str());
				falseResult++;
			}

		}
	}

	printf("정확도: (%d/%d) % \n", trueResult, (trueResult + falseResult));
}

/*

이미지 한개만 테스트하고 싶을 때

*/


void iconRecog::testOneData(string fileName, int iconIndex) {

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm = ml::SVM::load("trainedSVM.xml");

	Mat img, img_gray;
	img = imread(fileName);

	resize(img, img, Size(32, 32), 0, 0, CV_INTER_LANCZOS4);

	cvtColor(img, img_gray, CV_RGB2GRAY);

	HOGDescriptor d(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4), 9);

	vector<float> descriptorsValues;
	vector<Point> locations;
	d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

	int row = 1, col = descriptorsValues.size();

	Mat M(row, col, CV_32F);
	memcpy(&(M.data[0]), descriptorsValues.data(), col * sizeof(float));

	int result = svm->predict(M);

	if (result == iconIndex) {
		printf("%s --> %s [true!] \n", fileName.c_str(), iconClass[result].c_str());
	}
	else
		printf("%s --> %s [False!] \n", fileName.c_str(), iconClass[result].c_str());

}