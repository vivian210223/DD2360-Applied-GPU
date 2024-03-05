
#include <stdio.h>
#include <sys/time.h>

#define DataType double
// Define a small relative error threshold
const double relativeErrorThreshold = 1e-6;

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  int r = blockIdx.x*blockDim.x+threadIdx.x;
  int c = blockIdx.y*blockDim.y+threadIdx.y;
  //@@ Insert code to implement matrix multiplication here
  if (r<numARows && c<numBColumns){
    DataType tmp = 0;
    for (int k=0; k<numBRows; k++){
      tmp += A[r*numARows+k]*B[k*numBRows+c];
    }
    C[numARows*r+c] = tmp;  
  }
   
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[2]);
  numAColumns = atoi(argv[3]);
  numBRows = atoi(argv[4]);
  numBColumns = atoi(argv[5]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *) malloc(numARows*numAColumns * sizeof(DataType));
  hostB = (DataType *) malloc(numBRows*numBColumns * sizeof(DataType));
  hostC = (DataType *) malloc(numCRows*numCColumns * sizeof(DataType));
  resultRef = (DataType *) malloc(numCRows*numCColumns * sizeof(DataType));
  
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand(time(0)); 
   
  for (int i=0; i<numARows; i++){
    for (int j=0; j<numAColumns; j++){
      hostA[i*numARows+j] = rand();
    }     
  }
  
  for (int i=0; i<numBRows; i++){
    for (int j=0; j<numBColumns; j++){
      hostB[i*numBRows+j] = rand();
    }    
  }
     
  for (int i=0; i<numCRows; i++){
    for (int j=0; j<numCColumns; j++){
      DataType tmp = 0;
      for (int k=0; k<numBRows; k++){
        tmp += hostA[i*numAColumns+k]*hostB[k*numBColumns+j];
      }
      resultRef[i*numCColumns+j] = tmp;
    }     
  }
  
  
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows*numAColumns*sizeof(DataType));
  cudaMalloc(&deviceB, numBRows*numBColumns*sizeof(DataType));
  cudaMalloc(&deviceC, numCRows*numCColumns*sizeof(DataType));

  //create timer.
  clock_t H2D_1;
  H2D_1 = clock();
  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numCRows*numCColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  //cudaDeviceSynchronize();
  //stop timer.
  clock_t H2D_2;
  H2D_2 = clock();
  float H2D = H2D_2-H2D_1;
  

  //@@ Initialize the grid and block dimensions here
  dim3 TPB(atoi(argv[1]), atoi(argv[1]));
  int n = (numCRows*numCColumns + atoi(argv[1]) - 1)/atoi(argv[1]);
  dim3 BPG((numARows + atoi(argv[1]) - 1) / atoi(argv[1]), (numBColumns + atoi(argv[1]) - 1) / atoi(argv[1]));

  //dim3 BPG(n, n);
  
  //create timer.
  clock_t ker_1;
  ker_1 = clock();
  //@@ Launch the GPU Kernel here
  cudaDeviceSynchronize();
  gemm<<<BPG, TPB>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  //stop timer.
  clock_t ker_2;
  ker_2 = clock();
  float ker = ker_2-ker_1;

  //create timer.
  clock_t D2H_1;
  D2H_1 = clock();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  //stop timer.
  clock_t D2H_2;
  D2H_2 = clock();
  float D2H = D2H_2-D2H_1;
  
  //@@ Insert code below to compare the output with the reference
  int equal = 1;
  for (int i=0; i<numCRows; i++){
    for (int j=0; j<numCColumns; j++){
        DataType absDiff = fabs(resultRef[i * numCColumns + j] - hostC[i * numCColumns + j]);
        DataType absResult = fabs(resultRef[i * numCColumns + j]);
        DataType relativeError = absDiff / (absResult == 0.0 ? 1.0 : absResult);

        if (relativeError > relativeErrorThreshold) {
            printf("The results are not equal (%d,%d). ", i, j);
            printf("resultRef: %lf, hostC: %lf\n", resultRef[i * numCColumns + j], hostC[i * numCColumns + j]);
            equal = 0;
        }
    }     
  }
  if (equal){
    printf("The output of host are equal to reference\n");
  }
  
  printf("kernel:%f, H2D:%f, D2H:%f\n", ker, H2D, D2H);
  DataType sum = ker+H2D+D2H;
  printf("kernel:%f%%, H2D:%f%%, D2H:%f%%", ker/sum*100, H2D/sum*100, D2H/sum*100);
  

  
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
