/*

create by rosie
2017.03.30

*/
#include "iconRecog.h"


/*

OPEN CV HOG를 통해 각 아이콘의 edge 특징을 xml 파일로 추출

*/


void iconRecog::HOGfeature2XML() {

	char FullFileName[100];

	// ------------------------- positive data ----------------------------

	for (int i = 0; i < classifyNum; i++) {

		vector< vector < float> > v_descriptorsValues;
		vector< vector < Point> > v_locations;

		for (int j = 0; j < trainPosDataNum; j++) {

			sprintf_s(FullFileName, "%s%d.png", FileName[i].c_str(), j);

			Mat img, img_gray;
			img = imread(FullFileName);

			// --------------이미지 전처리-------------------

			resize(img, img, Size(24, 24), 0, 0, CV_INTER_LANCZOS4);

			cvtColor(img, img_gray, CV_RGB2GRAY);

			// ------------- 특징 추출 -----------------------

			HOGDescriptor d(Size(24, 24), Size(8, 8), Size(4, 4), Size(4, 4), 9);

			vector< float> descriptorsValues;
			vector< Point> locations;
			d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			v_descriptorsValues.push_back(descriptorsValues);
			v_locations.push_back(locations);


		}

		// -------------------- 특징 정보 xml 데이터로 저장 ----------------

		FileStorage hogXml(hogFileName[i], FileStorage::WRITE);

		int row = v_descriptorsValues.size(), col = v_descriptorsValues[0].size();

		Mat M(row, col, CV_32F);
		for (int i = 0; i < row; ++i)
			memcpy(&(M.data[col * i * sizeof(float)]), v_descriptorsValues[i].data(), col * sizeof(float));

		write(hogXml, "Descriptor_of_images", M);
		hogXml.release();

	}

	// ---------------------------------- Negative Data --------------------------------------------

	vector< vector < float> > v_NdescriptorsValues;
	vector< vector < Point> > v_Nlocations;

	for (int j = 0; j < trainNegDataNum ; j++) {

		sprintf_s(FullFileName, "%s%d.png", FileName[classifyNum].c_str(), j);

		Mat img, img_gray;
		img = imread(FullFileName);

		// --------------이미지 전처리-------------------

		resize(img, img, Size(24, 24), 0, 0, CV_INTER_LANCZOS4);

		cvtColor(img, img_gray, CV_RGB2GRAY);

		// ------------- 특징 추출 -----------------------

		HOGDescriptor d(Size(24, 24), Size(8, 8), Size(4, 4), Size(4, 4), 9);

		vector< float> descriptorsValues;
		vector< Point> locations;
		d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		v_NdescriptorsValues.push_back(descriptorsValues);
		v_Nlocations.push_back(locations);


	}

	FileStorage hogXml(hogFileName[classifyNum], FileStorage::WRITE);

	int row = v_NdescriptorsValues.size(), col = v_NdescriptorsValues[0].size();

	Mat M(row, col, CV_32F);
	for (int i = 0; i < row; ++i)
		memcpy(&(M.data[col * i * sizeof(float)]), v_NdescriptorsValues[i].data(), col * sizeof(float));

	write(hogXml, "Descriptor_of_images", M);
	hogXml.release();

	printf("HOGfeature2XML donw! \n");

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
	FileStorage read_NegativeXml(hogFileName[classifyNum], FileStorage::READ);
	//Negative Mat
	Mat nMat;
	read_NegativeXml["Descriptor_of_images"] >> nMat;

	int nRow, nCol;
	nRow = nMat.rows; nCol = nMat.cols;
	read_NegativeXml.release();

	for (int i = 0; i < classifyNum; i++) {

		FileStorage read_PositiveXml(hogFileName[i], FileStorage::READ);

		//Positive Mat
		Mat pMat;
		read_PositiveXml["Descriptor_of_images"] >> pMat;
		positiveMat.push_back(pMat);

		read_PositiveXml.release();

	}

	// ---------------- 여러 클래스 데이터를 하나의 Mat으로 통합------------------
	printf("2. Make training data for SVM\n");
	// rows == # of training data, cols == # of descriptor
	Mat trainigData(positiveMat[0].rows * classifyNum + nRow, nCol, CV_32FC1);
	// initialize label with last integer value
	Mat labels(positiveMat[0].rows * classifyNum + nRow, 1, CV_32S, Scalar(classifyNum));

	int startData = 0;
	int startLabel = 0;
	// input positive data set and label
	for (int i = 0; i < classifyNum; i++) {
		memcpy(&(trainigData.data[startData]), positiveMat[i].data, sizeof(float) * positiveMat[i].cols * positiveMat[i].rows);
		labels.rowRange(startLabel, startLabel + positiveMat[i].rows) = Scalar(i);
		startLabel = startLabel + positiveMat[i].rows;
		startData = startData + sizeof(float) * positiveMat[i].cols * positiveMat[i].rows; 
	}
	// input negative data set
	memcpy(&(trainigData.data[startData]), nMat.data, sizeof(float) * nCol * nRow);

	// ------------------- svm 모델 생성 및 트레이닝 후 모델을 xml 데이터로 저장 -------------------------------
	printf("3. Set SVM parameter \n");

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::POLY);
	svm->setDegree(3);
	svm->setGamma(2);
	svm->setCoef0(0);
	svm->setC(2);
//	svm->setNu(0.2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 1e-6));

	printf("4. Training \n");

	svm->train(trainigData, ml::ROW_SAMPLE, labels);

	printf("5. SVM XML save \n");
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

	for (int i = 0; i < classifyNum; i++) {

		for (int j = trainPosDataNum; j < totalPosDataNum ; j++) {

			sprintf_s(FullFileName, "%s%d.png", FileName[i].c_str(), j);

			Mat img, img_gray;
			img = imread(FullFileName);

			resize(img, img, Size(24, 24), 0, 0, CV_INTER_LANCZOS4);

			cvtColor(img, img_gray, CV_RGB2GRAY);

			HOGDescriptor d(Size(24, 24), Size(8, 8), Size(4, 4), Size(4, 4), 9);

			vector<float> descriptorsValues;
			vector<Point> locations;
			d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			// ----------Classification----------------

			int row = 1, col = descriptorsValues.size();

			Mat M(row, col, CV_32F);
			memcpy(&(M.data[0]), descriptorsValues.data(), col * sizeof(float));

			int result = svm->predict(M);

			if ( result == i) {
				printf("%s --> %s [true] \n", FullFileName, iconClass[result].c_str());
				trueResult++;
			}
			else if ( result != i) {
				printf("%s --> %s [false] \n", FullFileName, iconClass[result].c_str());
				falseResult++;
			}

		}
	}

	for (int j = trainNegDataNum ; j < totalNegDataNum; j++) {

		sprintf_s(FullFileName, "%s%d.png", FileName[classifyNum].c_str(), j);

		Mat img, img_gray;
		img = imread(FullFileName);

		resize(img, img, Size(24, 24), 0, 0, CV_INTER_LANCZOS4);

		cvtColor(img, img_gray, CV_RGB2GRAY);

		HOGDescriptor d(Size(24, 24), Size(8, 8), Size(4, 4), Size(4, 4), 9);

		vector<float> descriptorsValues;
		vector<Point> locations;
		d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		// ----------Classification----------------

		int row = 1, col = descriptorsValues.size();

		Mat M(row, col, CV_32F);
		memcpy(&(M.data[0]), descriptorsValues.data(), col * sizeof(float));

		int result = svm->predict(M);

		if ( result >= classifyNum ) {
			printf("%s --> %s [true] \n", FullFileName, iconClass[result].c_str());
			trueResult++;
		}
		else if ( result < classifyNum) {
			printf("%s --> %s [false] \n", FullFileName, iconClass[result].c_str());
			falseResult++;
		}

	}

	printf("Accuracy: ture vs. false(%d vs %d) --> %0.2f \% \n", trueResult, falseResult, (float) 100 * trueResult/(trueResult+falseResult) );
}

/*

이미지 한개만 테스트하고 싶을 때

*/
