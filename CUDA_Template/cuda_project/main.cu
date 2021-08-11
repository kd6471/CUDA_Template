#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "DS_timer.h"
#include "DS_definitions.h"

#define BLOCK_SIZE 256

using namespace cv;
using namespace std;

__global__ void cuda_ver1(int elem_size, int origin_total, int origin_cols, int origin_rows, int tem_cols, int tem_rows, uchar* ori_mPtr, uchar* tem_mPtr, int* r_x, int* r_y, int* count) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//tid가 origin의 크기를 벗어나는 것들은 하지 않음
	if (tid > origin_total)
		return;

	//해당스레드가 맡은 픽셀 (x,y)
	int x = tid / origin_cols;
	int y = tid % origin_cols;
	int ori_x;
	int ori_y;
	int ori_index;
	int tem_index;

	//해당 픽셀이 origin 크기 안에 있으면 실행
	if (x <= origin_cols - tem_cols || y <= origin_rows - tem_rows) {
		//해당 스레드가 맡은 origin 픽셀을 왼쪽 상단 시작점으로 하여 템플렛크기 만큼 서로 비교
		for (int a = 0; a < tem_cols; a++) {
			for (int b = 0; b < tem_rows; b++) {
				
				ori_index = x + a + (y + b) * origin_cols;
				tem_index = a + b * tem_cols;
					
				if ((ori_mPtr[ori_index * 3 + 2] != tem_mPtr[tem_index * 3 + 2])
					|| (ori_mPtr[ori_index * 3 + 1] != tem_mPtr[tem_index * 3 + 1])
					|| (ori_mPtr[ori_index * 3 + 0] != tem_mPtr[tem_index * 3 + 0])) {
					return;
				}
				
			}

		}

		//해당 스레드에서 비교한 tem와 origin의 tem크기만큼의 부분이 모두 같은 경우
			//해당 좌표 x,y를 *r_x와r_y에 값 변경
		int k = atomicAdd(count, 1);
		if (k == 0) {
			printf("x : %d\ty : %d\n", x, y);
			atomicAdd(r_x, x);
			atomicAdd(r_y, y);
		}
		
		
		
	}
	
}
__global__ void cuda_ver2(int origin_total, int origin_cols, int origin_rows, int tem_cols, int tem_rows, uchar* ori_mPtr, uchar* tem_mPtr, int* r_x, int* r_y, int* count) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid > origin_total)
		return;
	
	
	//해당스레드가 맡은 픽셀 (x,y)
	int x = tid / origin_cols;
	int y = tid % origin_cols;
	int ori_x;
	int ori_y;
	int ori_index;
	int tem_index;

	//해당 픽셀이 origin 크기 안에 있으면 실행
	if (x <= origin_cols - tem_cols || y <= origin_rows - tem_rows) {
		//해당 스레드가 맡은 origin 픽셀을 왼쪽 상단 시작점으로 하여 템플렛크기 만큼 서로 비교
		for (int a = 0; a < tem_cols; a++) {
			for (int b = 0; b < tem_rows; b++) {

				ori_index = x + a + (y + b) * origin_cols;
				tem_index = a + b * tem_cols;

				if (ori_mPtr[ori_index] != tem_mPtr[tem_index])
					return;	
			}
		}

		//해당 스레드에서 비교한 tem와 origin의 tem크기만큼의 부분이 모두 같은 경우
			//해당 좌표 x,y를 *r_x와r_y에 값 변경
		int k = atomicAdd(count, 1);
		if (k == 0) {
			printf("x : %d\ty : %d\n", x, y);
			atomicAdd(r_x, x);
			atomicAdd(r_y, y);
		}

	}
	
}

void setRed(Mat img, int x, int y, int tem_cols, int tem_rows);
void serial(char* argv1, char* argv2);
void gpu_v1(char* argv1, char* argv2);
void gpu_v2(char* argv1, char* argv2);
void saveResult(char* name, Mat img);
void printInfo(char* name, Mat img);

DS_timer timer(7);

void main(int argc, char* argv[])
{
	
	timer.setTimerName(0, (char*)"[Serial]");
	timer.setTimerName(1, (char*)"[Parallel CUDA V1 Host To Device]");
	timer.setTimerName(2, (char*)"[Parallel CUDA V1 Compute]");
	timer.setTimerName(3, (char*)"[Parallel CUDA V1 Device To Host]");
	timer.setTimerName(4, (char*)"[Parallel CUDA V2 Host To Device]");
	timer.setTimerName(5, (char*)"[Parallel CUDA V2 Compute]");
	timer.setTimerName(6, (char*)"[Parallel CUDA V2 Device To Host]");
	
	serial(argv[1], argv[2]);
	
	gpu_v1(argv[1], argv[2]);

	gpu_v2(argv[1], argv[2]);

	timer.printTimer();
}
// 버전별로 입력 데이터 정보 출력 함수
void printInfo(char* name,Mat img) { 
	printf("------------\n");
	printf("File Name : %s\n",name);
	printf("cols : %d\n", img.cols);
	printf("rows : %d\n", img.rows);
	printf("total : %d\n", img.total());
	printf("elemSize : %d\n", img.elemSize());
	printf("sizeInBytes : %d\n", img.total() * img.elemSize());
	printf("------------\n");
}
// 결과 이미지에 빨간색 사각형을 그리는 함수
void setRed(Mat img,int x, int y, int tem_cols,int tem_rows) {
for (int a = 0; a < tem_cols; a++) {
	for (int b = 0; b < tem_rows; b += tem_rows - 1) {
		int t_x = x + a;
		int t_y = y + b;

		int index = t_x + t_y * img.cols;
		img.at<Vec3b>(t_y, t_x)[0] = 0;
		img.at<Vec3b>(t_y, t_x)[1] = 0;
		img.at<Vec3b>(t_y, t_x)[2] = 255;
	}
}
for (int a = 0; a < tem_rows; a++) {
	for (int b = 0; b < tem_cols; b += tem_cols - 1) {
		int t_x = x + b;
		int t_y = y + a;

		int index = t_x + t_y * img.cols;
		img.at<Vec3b>(t_y, t_x)[0] = 0;
		img.at<Vec3b>(t_y, t_x)[1] = 0;
		img.at<Vec3b>(t_y, t_x)[2] = 255;
	}
}
}
// 결과 이미지에 빨간색 사각형을 포함해 새로 저장하는 함수
void saveResult(char* name, Mat img) {
	cv::imwrite(name, img);
}

void serial(char* argv1, char* argv2) {
	printf("----Serial-----\n");
	cv::Mat origin = cv::imread(argv1, cv::IMREAD_COLOR);
	size_t ori_sizeInBytes = origin.total() * origin.elemSize();
	printInfo(argv1, origin);

	uchar* ori_mPtr = new uchar[ori_sizeInBytes];
	std::memcpy(ori_mPtr, origin.data, ori_sizeInBytes);


	cv::Mat tem = cv::imread(argv2, cv::IMREAD_COLOR);
	size_t tem_sizeInBytes = tem.total() * tem.elemSize();
	printInfo(argv2, tem);

	uchar* tem_mPtr = new uchar[tem_sizeInBytes];
	std::memcpy(tem_mPtr, tem.data, tem_sizeInBytes);

	timer.onTimer(0);
	//Serial
	int r_x = -1;
	int r_y = -1;
	int elem_size = origin.elemSize();
	for (int i = 0; i < origin.total(); i++) {
		int x = i / origin.cols;
		int y = i % origin.cols;
		if (x > origin.cols - tem.cols || y > origin.rows - tem.rows)
			continue;

		bool result = true;
		for (int a = 0; a < tem.cols && result; a++) {
			for (int b = 0; b < tem.rows && result; b++) {
				int ori_x = x + a;
				int ori_y = y + b;
				int ori_index = ori_x + ori_y * origin.cols;
				int tem_index = a + b * tem.cols;

				for (int elem = 0; elem < elem_size; elem++) {
					if (ori_mPtr[ori_index * elem_size + elem] != tem_mPtr[tem_index * elem_size + elem]) {
					
						result = false;
						break;
					}
				}
			}
		}
		if (result == true) {
			r_x = x;
			r_y = y;
			break;
		}
	}

	if (r_x == -1 || r_y == -1) {
		printf("origin내 template를 찾을 수 없습니다.\n");
	}
	else {
		printf("성공 : %d,%d\n", r_x, r_y);
		setRed(origin, r_x, r_y, tem.cols, tem.rows);
	}
	timer.offTimer(0);
	saveResult("[result]serial.png", origin);
	origin.release();
	tem.release();
	free(ori_mPtr);
	free(tem_mPtr);
}

void gpu_v1(char* argv1, char* argv2) {
	printf("----GPU V1-----\n");

	cv::Mat origin = cv::imread(argv1, cv::IMREAD_COLOR);
	size_t ori_sizeInBytes = origin.total() * origin.elemSize();
	printInfo(argv1, origin);

	uchar* ori_mPtr = new uchar[ori_sizeInBytes];
	std::memcpy(ori_mPtr, origin.data, ori_sizeInBytes);


	cv::Mat tem = cv::imread(argv2, cv::IMREAD_COLOR);
	size_t tem_sizeInBytes = tem.total() * tem.elemSize();
	printInfo(argv2, tem);

	uchar* tem_mPtr = new uchar[tem_sizeInBytes];
	std::memcpy(tem_mPtr, tem.data, tem_sizeInBytes);

	int elem_size = origin.elemSize();
	int ori_total = origin.total();
	int ori_cols = origin.cols;
	int ori_rows = origin.rows;
	int tem_cols = tem.cols;
	int tem_rows = tem.rows;

	int x = 0;
	int y = 0;
	int cnt = 0;

	int* d_x;
	int* d_y;
	int* d_cnt;
	uchar* d_ori;
	uchar* d_tem;

	d_ori = new uchar[ori_sizeInBytes];
	d_tem = new uchar[tem_sizeInBytes];

	cudaMalloc(&d_ori, ori_sizeInBytes);
	cudaMalloc(&d_tem, tem_sizeInBytes);

	timer.onTimer(1);
	cudaMemcpy(d_tem, tem_mPtr, tem_sizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori, ori_mPtr, ori_sizeInBytes, cudaMemcpyHostToDevice);
	timer.offTimer(1);
	cudaMalloc(&d_x, sizeof(int)); cudaMemset(d_x, 0, sizeof(int));
	cudaMalloc(&d_y, sizeof(int)); cudaMemset(d_y, 0, sizeof(int));
	cudaMalloc(&d_cnt, sizeof(int)); cudaMemset(d_cnt, 0, sizeof(int));

	dim3 dimGrid(ceil((double)ori_total / BLOCK_SIZE), 1, 1);
	dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
	timer.onTimer(2);
	cuda_ver1 << < dimGrid, dimBlock >> > (elem_size, ori_total, ori_cols, ori_rows, tem_cols, tem_rows, d_ori, d_tem, d_x, d_y, d_cnt);
	cudaDeviceSynchronize(); // synchronization function
	timer.offTimer(2);
	//포인트 값을 다시 CPU로 가져온다. 

	timer.onTimer(3);
	cudaMemcpy(&x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&y, d_y, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
	timer.offTimer(3);
	printf("GPU 일치 부분 왼쪽 상단 좌표(%d) : %d, %d\n", cnt, x, y);


	//해당 좌표 출력
	if (cnt==0) {
		printf("origin내 template를 찾을 수 없습니다.\n");
	}
	else {
		printf("성공 : %d,%d\n", x, y);
		setRed(origin, x, y, tem.cols, tem.rows);
	}
	saveResult("[result]gpu_v1.png", origin);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_cnt);
	cudaFree(d_ori);
	cudaFree(d_tem);
	delete[] ori_mPtr;
	delete[] tem_mPtr;
	origin.release();
	tem.release();
	printf("---------------\n");
}

void gpu_v2(char* argv1, char* argv2) {
	printf("----GPU V2-----\n");

	cv::Mat origin = cv::imread(argv1, cv::IMREAD_GRAYSCALE);
	size_t ori_sizeInBytes = origin.total() * origin.elemSize();
	printInfo(argv1, origin);

	uchar* ori_mPtr = new uchar[ori_sizeInBytes];
	std::memcpy(ori_mPtr, origin.data, ori_sizeInBytes);


	cv::Mat tem = cv::imread(argv2, cv::IMREAD_GRAYSCALE);
	size_t tem_sizeInBytes = tem.total() * tem.elemSize();
	printInfo(argv2, tem);

	uchar* tem_mPtr = new uchar[tem_sizeInBytes];
	std::memcpy(tem_mPtr, tem.data, tem_sizeInBytes);
	
	int ori_total = origin.total();
	int ori_cols = origin.cols;
	int ori_rows = origin.rows;
	int tem_cols = tem.cols;
	int tem_rows = tem.rows;

	int x = 0;
	int y = 0;
	int cnt = 0;

	int* d_x;
	int* d_y;
	int* d_cnt;
	uchar* d_ori;
	uchar* d_tem;

	d_ori = new uchar[ori_sizeInBytes];
	d_tem = new uchar[tem_sizeInBytes];

	cudaMalloc(&d_ori, ori_sizeInBytes);
	cudaMalloc(&d_tem, tem_sizeInBytes);

	timer.onTimer(4);
	cudaMemcpy(d_tem, tem_mPtr, tem_sizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori, ori_mPtr, ori_sizeInBytes, cudaMemcpyHostToDevice);
	timer.offTimer(4);

	cudaMalloc(&d_x, sizeof(int)); cudaMemset(d_x, 0, sizeof(int));
	cudaMalloc(&d_y, sizeof(int)); cudaMemset(d_y, 0, sizeof(int));
	cudaMalloc(&d_cnt, sizeof(int)); cudaMemset(d_cnt, 0, sizeof(int));

	dim3 dimGrid(ceil((double)ori_total / BLOCK_SIZE), 1, 1);
	dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
	timer.onTimer(5);
	cuda_ver2 << < dimGrid, dimBlock >> > (ori_total, ori_cols, ori_rows, tem_cols, tem_rows, d_ori, d_tem, d_x, d_y, d_cnt);
	cudaDeviceSynchronize(); // synchronization function
	timer.offTimer(5);
	//포인트 값을 다시 CPU로 가져온다. 
	timer.onTimer(6);
	cudaMemcpy(&x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&y, d_y, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
	timer.offTimer(6);
	printf("GPU 일치 부분 왼쪽 상단 좌표(%d) : %d, %d\n", cnt, x, y);
	
	origin.release();
	origin = cv::imread(argv1, cv::IMREAD_COLOR);

	//해당 좌표 출력
	if (cnt == 0) {
		printf("origin내 template를 찾을 수 없습니다.\n");
	}
	else {
		printf("성공 : %d,%d\n", x, y);
		setRed(origin, x, y, tem.cols, tem.rows);

		
		//cv::imshow("img", result);

		//cv::waitKey(10000);
	}
	saveResult("[result]gpu_v2.png", origin);


	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_cnt);
	cudaFree(d_ori);
	cudaFree(d_tem);
	
	delete[] ori_mPtr;
	delete[] tem_mPtr;
	origin.release();
	tem.release();
	printf("---------------\n");
}
