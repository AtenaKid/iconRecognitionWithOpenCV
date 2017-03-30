#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

static int SUM_MODE = 0;
    
int classifyNum = 8;
int trainDataNum = 35;
int totalDataNum = 40;

int backgroundNum = 27;

static vector<String> saveHogDesFileName = {
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


static vector<String> iconClass = {
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

static vector<String> FirstFileName = {
	"./images/close/close",
	"./images/back/back",
	"./images/home/home",
	"./images/menu/menu",
	"./images/profile/profile",
	"./images/search/search",
	"./images/settings/settings",
	"./images/shopping/shopping",
	"./images/negative/negative" };



void addDataSet(int mode);
void HOGfeature2XML();
void trainingBySVM();
void testSVMTrainedData();
void testOneData(String filename, int iconIndex);

int main() {

//  addDataSet(SUM_MODE);
//	HOGfeature2XML();
//	trainingBySVM();
	testSVMTrainedData();

	return 0;

}

void addDataSet(int mode) {

	// �̹��� �������� ���鼭 ����
	for (int i = 0; i < classifyNum + 1; ++i) {

		char FullFileName[100];
		// �� �̹��� ���� ���鼭 ����
		for (int j = 0; j < trainDataNum; ++j) {

			// �̹��� ���� �̸�
			sprintf_s(FullFileName, "%s%d.png", FirstFileName[i].c_str(), j);

			printf("%s loaded! \n", FullFileName);

			Mat img = imread(FullFileName);
			
			if (mode == SUM_MODE) {

				char backImageName[100];
				char NewFileName[100];
				for (int k = 0; k < backgroundNum; k++) {
					// ��� �̹����� �޾ƿ�
					sprintf_s(backImageName, "./images/background/background%d.png", k);
					printf("%s loaded! \n", backImageName);
					Mat backimg = imread(backImageName);

					// �� �̹����� ����� �����ϵ��� ����
					resize(img, img, Size(64, 64), 0, 0, CV_INTER_LANCZOS4);
					resize(backimg, backimg, Size(64, 64), 0, 0, CV_INTER_LANCZOS4);

					Mat img_gray, mask, mask_inv;
					// �������� �׷��� �����Ϸ� ��ȯ
					cvtColor(img, img_gray, CV_RGB2GRAY);

					// �׷��� ������ �޾Ƽ� �̹����� ����ȭ�� (200 �̻� �������)
					threshold(img_gray, mask, 200, 255, THRESH_BINARY); 
					// 1-> 0 , 0 ->1�� ����
					bitwise_not(mask, mask_inv);

					// ��¦ �β��� (�����ϰ�)
					Mat kernal = Mat::ones(2, 2, CV_8U);
					dilate(mask_inv, mask_inv, kernal);

					Mat img_bg, img_fg; 
					bitwise_and(backimg, backimg, img_bg, mask = mask);
					bitwise_and(img, img, img_fg, mask = mask_inv );

					Mat sumImage;
					add(img_bg, img_fg, sumImage);
					sprintf_s(NewFileName, "%s%d_M%d.png", FirstFileName[i].c_str(), j , k);
					imwrite(NewFileName, sumImage);
				}

			}
		}
	}
}


void HOGfeature2XML() {

	char FullFileName[100];

	// �̹��� �������� ���鼭 ����
	for (int i = 0; i < (classifyNum+1); ++i) {

		vector< vector < float> > v_descriptorsValues;
		vector< vector < Point> > v_locations;

		// �� �̹��� ���� ���鼭 ����
		for (int j = 0; j < trainDataNum ; ++j) {

			// �̹��� ���� �̸�
			sprintf_s(FullFileName, "%s%d.png", FirstFileName[i].c_str(), j);

			printf("%s loaded! \n", FullFileName);

			// �̹����� �о����
			Mat img, img_gray;
			img = imread(FullFileName);

			// --------------�̹��� ��ó��-------------------
			// �̹��� �������� (���ʽ� �˰��� ���)
			resize(img, img, Size(32, 32), 0, 0, CV_INTER_LANCZOS4);

			// �׷��� �����Ϸ� ����
			cvtColor(img, img_gray, CV_RGB2GRAY);

			// ------------- Ư¡ ���� -----------------------

			// edge ���⼺ ���� --> ������׷� --> bin�� �Ϸķ� ������ ����  Ư¡ ���� win size, block size, block stride, cell size
			HOGDescriptor d(Size(32, 32), Size(8, 8), Size(4, 4), Size(4, 4), 9);

			vector< float> descriptorsValues;
			vector< Point> locations;
			d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			// ���͸� ���ο� �ִ� ��� �̹��� Ư¡ ������ �ϳ��� ���Ϳ� ����(SVM)
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

		// -------------------- Ư¡ ���� xml �����ͷ� ���� ----------------

		FileStorage hogXml(saveHogDesFileName[i], FileStorage::WRITE); // opneCV���� �����ϴ� ���� ����� �Լ�

		int row = v_descriptorsValues.size(), col = v_descriptorsValues[0].size();

		Mat M(row, col, CV_32F);
		for (int i = 0; i < row; ++i)
			// Mat�� �����ʹ� 1d �������� �Ǿ� �����Ƿ� �� ���Ŀ� �°� ���� ���� (row(���� 1��) ���� ��)
			memcpy(&(M.data[col * i * sizeof(float)]), v_descriptorsValues[i].data(), col * sizeof(float));

		write(hogXml, "Descriptor_of_images", M);
		hogXml.release();

	}

}



void trainingBySVM() {

	vector<Mat> positiveMat;
	// ------------- Ư¡ ������ ��� xml ������ Mat���� �о� ���� ---------------
	printf("1. feature data load \n");
	printf("%s feature data load \n", saveHogDesFileName[classifyNum].c_str());
	FileStorage read_NegativeXml(saveHogDesFileName[classifyNum], FileStorage::READ);
	//Negative Mat
	Mat nMat;
	read_NegativeXml["Descriptor_of_images"] >> nMat;
	//Read Row, Cols
	int nRow, nCol;
	nRow = nMat.rows; nCol = nMat.cols;
	printf("rows: %d , cols: %d \n", nRow, nCol);

	//release
	read_NegativeXml.release();

	for (int i = 0; i < (int) saveHogDesFileName.size() - 1; ++i){

		printf("%s feature data load \n", saveHogDesFileName[i].c_str());

		FileStorage read_PositiveXml(saveHogDesFileName[i], FileStorage::READ);

		//Positive Mat
		Mat pMat;
		read_PositiveXml["Descriptor_of_images"] >> pMat;
		positiveMat.push_back(pMat);
		printf("rows: %d , cols: %d \n", positiveMat[i].rows, positiveMat[i].cols);

		//release
		read_PositiveXml.release();

	}

	// ---------------- ���� Ŭ���� �����͸� �ϳ��� Mat���� ����------------------
	printf("2. Make training data for SVM\n");
	//descriptor data set
	// rows == # of training data, cols == # of descriptor
	Mat trainigData(positiveMat[0].rows * classifyNum + nRow, nCol, CV_32FC1); 
	// �� ��ǲ�� ���� �󺧸� 0���� �ʱ�ȭ
	Mat labels(nRow*(classifyNum+1), 1, CV_32S, Scalar(classifyNum));
	
	printf("label row: %d, cols: %d \n", labels.rows, labels.cols);

	// positive data ���� ����
	int startData = 0;
	int startLabel= 0;
	for (int i = 0; i < (int)positiveMat.size() ; ++i ) {
		memcpy(&(trainigData.data[startData]), positiveMat[i].data, sizeof(float) * positiveMat[i].cols * nRow);
		labels.rowRange(startLabel, startLabel + nRow) = Scalar(i);
		startLabel = startLabel + nRow;
		startData = startData + sizeof(float) * positiveMat[i].cols * nRow; // ������ ����ų ��ġ
	}
	// negative data ���� ����
	memcpy(&(trainigData.data[startData]), nMat.data, sizeof(float) * nMat.cols * nMat.rows); 
	

	// ------------------- svm �� ���� �� Ʈ���̴� �� ���� xml �����ͷ� ���� -------------------------------
	printf("3. Set SVM parameter \n");

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm->setType(ml::SVM::NU_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setGamma(1);
	svm->setCoef0(0);
	svm->setNu(0.2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));

	printf("4. Training \n");
	
   	svm->train(trainigData,ml::ROW_SAMPLE,labels);

	printf("5. SVM XML svae \n");
	svm->save("trainedSVM.xml");

}

void testSVMTrainedData() {

	int trueResult = 0, falseResult = 0;

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm = ml::SVM::load("trainedSVM.xml");

	char FullFileName[100];

	// -------------- �׽�Ʈ �����͸� �о���̸鼭 ��� ���� -----------------------------

	for (int i = 0; i <= classifyNum; ++i) {

		for (int j = trainDataNum; j < totalDataNum ; ++j) {

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

	printf("��Ȯ��: (%d/%d) % \n", trueResult, (trueResult + falseResult));
}

void testOneData(String fileName, int iconIndex) {

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