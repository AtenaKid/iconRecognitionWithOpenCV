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

			cvtColor(img, img_gray, CV_RGB2GRAY);
			
			img_gray = crop(img_gray);
			img_gray = squalize(img_gray);

			resize(img_gray, img_gray, Size(WIN, WIN), 0, 0, CV_INTER_LANCZOS4);
		
			imshow("a", img_gray);
			waitKey(0);

			// ------------- 특징 추출 -----------------------

			HOGDescriptor d(Size(WIN, WIN), Size(BLOCK, BLOCK), Size(STRIDE, STRIDE), Size(CELL, CELL), BIN);

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

		cvtColor(img, img_gray, CV_RGB2GRAY);
		img_gray = crop(img_gray);
		img_gray = squalize(img_gray);
		resize(img_gray, img_gray, Size(WIN, WIN), 0, 0, CV_INTER_LANCZOS4);

		// ------------- 특징 추출 -----------------------

		HOGDescriptor d(Size(WIN, WIN), Size(BLOCK, BLOCK), Size(STRIDE, STRIDE), Size(CELL, CELL), BIN);

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
	svm->setDegree(4);
	svm->setGamma(4);
	svm->setCoef0(0);
	svm->setC(300);
//	svm->setNu(0.2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));

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
			
			cvtColor(img, img_gray, CV_RGB2GRAY);
			img_gray = crop(img_gray);
			img_gray = squalize(img_gray);

			resize(img_gray, img_gray, Size(WIN, WIN), 0, 0, CV_INTER_LANCZOS4);

			HOGDescriptor d(Size(WIN, WIN), Size(BLOCK, BLOCK), Size(STRIDE, STRIDE), Size(CELL, CELL), BIN); // 40도 단위, 셀사이즈 4X4

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
				Mat mat = getHogDescriptorVisual(img_gray, descriptorsValues, Size(WIN, WIN));
				char name[100];
				sprintf_s(name, "%s%d_%s.png", FileName[i].c_str(), j, iconClass[result].c_str());
				imwrite(name, mat);
			}

		}
	}

	for (int j = trainNegDataNum ; j < totalNegDataNum; j++) {

		sprintf_s(FullFileName, "%s%d.png", FileName[classifyNum].c_str(), j);

		Mat img, img_gray;
		img = imread(FullFileName);

		cvtColor(img, img_gray, CV_RGB2GRAY);

		img_gray = crop(img_gray);
		img_gray = squalize(img_gray);

		resize(img_gray, img_gray, Size(WIN, WIN), 0, 0, CV_INTER_LANCZOS4);

		HOGDescriptor d(Size(WIN, WIN), Size(BLOCK, BLOCK), Size(STRIDE, STRIDE), Size(CELL, CELL), BIN);

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
			Mat mat = getHogDescriptorVisual(img_gray, descriptorsValues, Size(WIN, WIN));
			char name[100];
			sprintf_s(name, "%s%d_%s.png", FileName[classifyNum].c_str(), j, iconClass[result].c_str());
			imwrite(name, mat);
		}

	}

	printf("Accuracy: ture vs. false(%d vs %d) --> %0.2f \% \n", trueResult, falseResult, (float) 100 * trueResult/(trueResult+falseResult) );
}




void iconRecog::testWithRealData() {

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm = ml::SVM::load("trainedSVM.xml");

	// -------------- 테스트 데이터를 읽어들이면서 결과 예측 -----------------------------

	
	for (int i = 0; i < testFileName.size(); i++) {

		Mat img, img_gray;
		img = imread(testFileName[i]);

		cvtColor(img, img_gray, CV_RGB2GRAY);

		img_gray = crop(img_gray);
		img_gray = squalize(img_gray);

		resize(img_gray, img_gray, Size(WIN, WIN), 0, 0, CV_INTER_LANCZOS4);

	
		HOGDescriptor d(Size(WIN, WIN), Size(BLOCK, BLOCK), Size(STRIDE, STRIDE), Size(CELL, CELL), BIN); // 40도 단위, 셀사이즈 4X4

		vector<float> descriptorsValues;
		vector<Point> locations;
		d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		// ----------Classification----------------

		int row = 1, col = descriptorsValues.size();

		Mat M(row, col, CV_32F);
		memcpy(&(M.data[0]), descriptorsValues.data(), col * sizeof(float));

		int result = svm->predict(M);

		//Mat hog = getHogDescriptorVisual(img_gray, descriptorsValues, Size(WIN, WIN));
		//imshow("a", hog);
		//waitKey(0);

		printf("%s --> %s \n", testFileName[i].c_str(), iconClass[result].c_str());
	
	}

}





Mat iconRecog::getHogDescriptorVisual(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = CELL;
	int gradientBinSize = BIN;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu


Mat iconRecog::squalize(Mat originMat) {

	int col = originMat.cols;
	int row = originMat.rows;

	if (col == row) {
		return originMat;
	}
	
//	uchar *data = originMat.data;
//	uchar *b, *g, *r;

	// (0,0)의 색상 추출
//	b = originMat.data + 0;
//	g = originMat.data + 1;
//	r = originMat.data + 2;

	int move1, move2;
	int margin = 8;
	Mat stride1, stride2, dst;
	
	if (col > row) {

		move1 = (int)((col - row + margin) / 2);
		move2 = (int)margin / 2;
		stride1 = (Mat_<float>(2, 3) << 1, 0, 0, 0, 1, move1); // move to down
		stride2 = (Mat_<float>(2, 3) << 1, 0, move2, 0, 1, 0); // move to right

		dst = Mat::zeros(Size(col+margin, col+margin), originMat.type());
		
	}else if (row > col) {

		move1 = (int)((row - col + margin) / 2);
		move2 = (int)margin / 2;

		stride1 = (Mat_<float>(2, 3) << 1, 0, move1, 0, 1, 0); // move to right
		stride2 = (Mat_<float>(2, 3) << 1, 0, 0, 0, 1, move2); // move to down

		dst = Mat::zeros(Size(row+margin, row+margin), originMat.type());

	}

	//warpAffine(originMat, dst, stride, dst.size(), INTER_LANCZOS4, BORDER_CONSTANT, cv::Scalar((int)*b, (int)*g, (int)*r));
	warpAffine(originMat, dst, stride1, dst.size(), INTER_LANCZOS4, BORDER_CONSTANT, cv::Scalar(background));
	warpAffine(dst, dst, stride2, dst.size(), INTER_LANCZOS4, BORDER_CONSTANT, cv::Scalar(background));
	
	return dst;

}

Mat iconRecog::crop(Mat originMat) {

	// originMat is grayScale
	uchar *data = originMat.data;
	vector<pair<int, int>> firstIndex;
	
	int row = originMat.rows;
	int col = originMat.cols;
	int threshold = 30;
	
	int current, before;

	//////////// left scanning /////////////
	for (int i = 0; i < row; i++) {

		for (int j = 0; j < col; j++) {
			if (j == 0) {
				before = (int)data[i*col + j];
			}
			current = (int) data[i*col + j];
			
			if (abs(current - before) > threshold) {
	
				int r = i;
				int c = j;
				
				background = before; // squalize background 초기화

				firstIndex.push_back({ r,c });
				
				break;
			}

			before = current;

		}

	}

	// find min col index
	int minCol = col;
	int minRow = row;

	for (int i = 0; i < firstIndex.size(); i++) {

		int r = firstIndex.at(i).first;
		int c = firstIndex.at(i).second;

		if (c < minCol) {
			minCol = c;
			minRow = r;
		}
	}

	int leftCol = minCol;

	firstIndex.clear();

	///////////// top scanning //////////////

	for (int i = 0; i < col; i++) {

		for (int j = 0; j < row; j++) {
			
			if (j == 0) {
				before = (int)data[j*col + i];
			}

			current = (int) data[j*col + i];

			if (abs(current - before) > threshold) {
				
				int r = (int) (j*col + i) / col;
				int c = (int)(j*col + i) % col;

				firstIndex.push_back({ r, c });
				break;
			}

			before = current;
		}

	}

	// find min row index
	
	minRow = row;
	minCol = col;

	for (int i = 0; i < firstIndex.size(); i++) {

		int r = firstIndex.at(i).first;
		int c = firstIndex.at(i).second;

		if (r < minRow) {
			minRow = r;
			minCol = c;
		}

	}

	int leftRow = minRow;

	firstIndex.clear();

	////////////// right scanning //////////////

	for (int i = 0; i < row; i++) {

		for (int j = col-1; j >= 0 ; j--) {
			if (j == col-1) {
				before = (int)data[i*col + j];
			}
			current = (int) data[i*col + j];

			if (abs(current - before) > threshold) {
				
				int r = (int)(i*col + j) / col;
				int c = (int)(i*col + j) % col;

				firstIndex.push_back({ r, c });

				break;
			}

			before = current;
		}

	}

	// find max col index

	int maxCol = 0;
	int maxRow = 0;

	for (int i = 0; i < firstIndex.size(); i++) {

		int r = firstIndex.at(i).first;
		int c = firstIndex.at(i).second;

		if (maxCol < c) {
			maxCol = c;
			maxRow = r;
		}
	}

	int rightCol = maxCol;

	firstIndex.clear();

	////////////// bottom scannig //////////////

	for (int i = 0; i < col; i++) {


		for (int j = row-1; j >= 0 ; j--) {
			if (j == row-1) {
				before = (int)data[col*j + i];
			}

			current = (int) data[col*j + i];

			if (abs(current - before) > threshold) {
				
				int r = (int)(col*j + i) / col;
				int c = (int)(col*j + i) % col;

				firstIndex.push_back({ r, c });
				break;
			}

			before = current;
		}

	}

	// find max row index
	
	maxRow = 0;
	maxCol = 0;

	for (int i = 0; i < firstIndex.size(); i++) {

		int r = firstIndex.at(i).first;
		int c = firstIndex.at(i).second;


		if (maxRow < r) {
			maxRow = r;
			maxCol = c;
		}
	}

	int rightRow = maxRow;

	// 관심영역 설정 (set ROI (X, Y, W, H)).

	if (leftCol == col || leftRow == row || rightCol - leftCol <= 0 || rightRow - leftRow <= 0 ) {
		printf("cannot cropping\n");
		return originMat;
	}

	Rect rect;
	rect = Rect(leftCol, leftRow, rightCol - leftCol, rightRow - leftRow);

	Mat cropMat = originMat(rect);
	
	return cropMat;

}



