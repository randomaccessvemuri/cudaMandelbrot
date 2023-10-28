#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>

typedef struct RGB_{
    int r;
    int g;
    int b;
} RGB;

typedef struct Coords_{
    float x;
    float y;
} Coords;

typedef struct imageIndexAndRGB_{
    int x;
    int y;
    RGB color;
} imageIndexAndRGB;



/// complexNumber abstracts away the functioning of a complex number to clean up the code later on, nothing new here; just standard stuff here!
class complexNumber{
private:
    float real;
    float imaginary;
public:
    __device__ __host__ complexNumber(float real, float imag){
        this->real = real;
        this->imaginary = imag;
    }

    __host__ __device__ complexNumber operator+(const complexNumber& complexIn){
        return complexNumber(this->real+complexIn.real, this->imaginary+complexIn.imaginary);
    }

    __host__ __device__ complexNumber operator*(const complexNumber& complexIn){
        return complexNumber(complexIn.real*this->real - complexIn.imaginary*this->imaginary , this->real*complexIn.imaginary + this->imaginary*complexIn.real);
    }

    __host__ __device__ float getReal(){
        return this->real;
    }

    __host__ __device__ float getImag(){
        return this->imaginary;
    }


};

class Image{
private:
    cv::Mat image;
public:
    Image(int width, int height){
        this->image = cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));
    }

    void setPixel(int x, int y, RGB color){
        this->image.at<cv::Vec3b>(y,x)[0] = color.b;
        this->image.at<cv::Vec3b>(y,x)[1] = color.g;
        this->image.at<cv::Vec3b>(y,x)[2] = color.r;
    }

    void clearImage(){
        this->image = cv::Mat(this->image.rows, this->image.cols, CV_8UC3, cv::Scalar(0,0,0));
    }

    void saveImage(std::string filename){
        cv::imwrite(filename, this->image);
    }

    void showImage(std::string windowName){
        cv::imshow(windowName, this->image);
        cv::waitKey(0);
    }
};

///This function computes the number of bounces until the "ray", so to speak, exits the screen. Has been written separate from the kernel to keep the code clean!
///@param [in] maxX The bounds of the screen in X-axis converted from PS to CS
///@param [in] maxY The bounds of the screen in Y-axis converted from PS to CS
///@param [in] x Current position converted from PS to CS
///@return An integer that describes the number of bounces to exit the screen (or the max bounces, whichever comes earlier!)
///@note PS to CS: Pixel space to Cartesian space
__device__ __host__ int computeBouncesMandelbrot(Coords actualXY, Coords MaxXY, int MAX_BOUNCES){

    complexNumber startPoint = complexNumber(actualXY.x,actualXY.y);
    complexNumber currentPoint = startPoint;
    int bounces = 0;

    /*The terminating condition is as such:
     *  if the real part has exceeded maxX (or the point has gone out of the screen on the X-axis)
     *  if the imaginary part has exceeded maxY (or the point has gone out of the screen on the Y-axis)
     *  if we have exceeded the max number of bounces, for the sake of keeping computation time sane, we cut off the computation!
    */
    while(currentPoint.getReal() < MaxXY.x && currentPoint.getImag() < MaxXY.y && bounces < MAX_BOUNCES){
        currentPoint = (currentPoint * currentPoint) + startPoint;
        bounces++;
    }
    return bounces;
}



__global__ void doBounces(imageIndexAndRGB* imageBuffer, float aspectRatio, int screenWidth, int screenHeight, float maxBoundsX, float maxBoundsY, float minBoundsX, float minBoundsY, int MAX_BOUNCES, float UOFFSET, float VOFFSET){
    //Obtain index using CUDA Intrinsics
    unsigned int PixelX = (blockIdx.x*blockDim.x) + threadIdx.x;
    unsigned int PixelY = (blockIdx.y*blockDim.y) + threadIdx.y;

    float FigureWidth = abs(maxBoundsX) + abs(minBoundsX);
    float FigureHeight = abs(maxBoundsY) + abs(minBoundsY);

    float PixelWidth = FigureWidth/screenWidth;
    float PixelHeight = FigureHeight/screenHeight;

    float cartesianPositionX = (PixelX*PixelWidth) + minBoundsX + UOFFSET;
    float cartesianPositionY = (PixelY*PixelHeight) + minBoundsY + VOFFSET;

    int bounces = computeBouncesMandelbrot({cartesianPositionX, cartesianPositionY}, {maxBoundsX, maxBoundsY}, MAX_BOUNCES);

    //printf("DEBUG INFO: Bounces for pixel (%d,%d), (%f,%f) is %d. Bounds are (%f,%f) to (%f,%f)\n", PixelX, PixelY, cartesianPositionX, cartesianPositionY, bounces, minBoundsX, minBoundsY, maxBoundsX, maxBoundsY);



    //Write to the image buffer
    if (bounces == MAX_BOUNCES){
        bounces = 0;
    }
    RGB color = {
            (bounces*4)%255,
            (bounces*8)%255,
            (bounces*16)%255
    };
    //OpenCV, for whatever reason, messes up conversion of 1D array to an image so we'll do that manually later!
    //For now, this suffices
    imageIndexAndRGB imageIndexAndRGB_d = {
            (int)PixelX,
            (int)PixelY,
            color
    };

    imageBuffer[PixelX*screenHeight + PixelY] = imageIndexAndRGB_d;

}

__host__ void D1BufferToImage(Image* imageIn, imageIndexAndRGB* imageBuffer, float aspectRatio, int SCREEN_WIDTH){
    int screenHeight = SCREEN_WIDTH/aspectRatio;
    int counter = 0;
    for (int i = 0; i < SCREEN_WIDTH; i++){
        for (int j = 0; j < screenHeight; j++){
            //std::cout<<"DEBUG INFO: Writing pixel ("<<i<<","<<j<<") to image with color ("<<imageBuffer[counter].color.r<<","<<imageBuffer[counter].color.g<<","<<imageBuffer[counter].color.b<<")"<<std::endl;
            imageIn->setPixel(i,j,imageBuffer[counter++].color);
        }
    }
}

int main(int argc, char** argv) {
    //Get parameters from command line
    int SCREEN_WIDTH;
    int MAX_BOUNCES;
    float MIN_BOUNDS_X;
    float MIN_BOUNDS_Y;
    float MAX_BOUNDS_X;
    float MAX_BOUNDS_Y;
    float UOFFSET;
    float VOFFSET;


    try{
        if(argc != 9){
            throw std::invalid_argument("Invalid number of arguments!");
        }
        SCREEN_WIDTH = std::stoi(argv[1]);
        MAX_BOUNCES = std::stoi(argv[2]);
        MIN_BOUNDS_X = std::stof(argv[3]);
        MIN_BOUNDS_Y = std::stof(argv[4]);
        MAX_BOUNDS_X = std::stof(argv[5]);
        MAX_BOUNDS_Y = std::stof(argv[6]);
        UOFFSET = std::stof(argv[7]);
        VOFFSET = std::stof(argv[8]);
    } catch (std::invalid_argument& e){
        std::cout << "Invalid argument: " << e.what() << std::endl;
        std::cout <<"USAGE: ./Mandelbrot <SCREEN_WIDTH> <MAX_BOUNCES> <MIN_BOUNDS_X> <MIN_BOUNDS_Y> <MAX_BOUNDS_X> <MAX_BOUNDS_Y> <UOFFSET> <VOFFSET>" << std::endl;
        return 1;
    }




    // Initialize the image
    float aspectRatio = (MAX_BOUNDS_X+(std::abs(MIN_BOUNDS_X)))/(MAX_BOUNDS_Y+(std::abs(MIN_BOUNDS_Y)));

    // Initialize the image with the correct dimensions
    int imageHeight = SCREEN_WIDTH / aspectRatio;
    Image* image = new Image(SCREEN_WIDTH, imageHeight);



    // Test on CPU

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i = 0; i < SCREEN_WIDTH; i++) {
        for (int j = 0; j < imageHeight; j++) {
            float cartesianPositionX = ((float) i / (float) SCREEN_WIDTH) * (MAX_BOUNDS_X - MIN_BOUNDS_X) + MIN_BOUNDS_X + UOFFSET;
            float cartesianPositionY = ((float) j / (float) imageHeight) * (MAX_BOUNDS_Y - MIN_BOUNDS_Y) + MIN_BOUNDS_Y + VOFFSET;
            int bounces = computeBouncesMandelbrot({cartesianPositionX, cartesianPositionY},
                                                   {(float) MAX_BOUNDS_X, (float) MAX_BOUNDS_Y}, MAX_BOUNCES);
            // You need to map bounces to a color here.
            if (bounces == MAX_BOUNCES) {
                bounces = 0;
            }
            RGB color = {
                    (bounces * 4) % 255,
                    (bounces * 2) % 255,
                    (bounces * 8) % 255
            };
            image->setPixel(i, j, color);

        }
    }
    image->showImage("CPU");

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    std::cout << "CPU time: " << time << " ms" << std::endl;

    // Test on GPU
    // Initialize the image with the correct dimensions
    imageIndexAndRGB * imageBufferGPU;
    cudaMallocManaged(&imageBufferGPU, SCREEN_WIDTH * imageHeight * sizeof(imageIndexAndRGB));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(SCREEN_WIDTH / threadsPerBlock.x, imageHeight / threadsPerBlock.y);

    image->clearImage();

    begin = std::chrono::steady_clock::now();


    doBounces<<<numBlocks, threadsPerBlock>>>(imageBufferGPU, aspectRatio, SCREEN_WIDTH, SCREEN_WIDTH/aspectRatio, MAX_BOUNDS_X, MAX_BOUNDS_Y, MIN_BOUNDS_X, MIN_BOUNDS_Y, MAX_BOUNCES, UOFFSET, VOFFSET);

    cudaDeviceSynchronize();

    time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    std::cout << "GPU time: " << time << " ms" << std::endl;

    D1BufferToImage(image, imageBufferGPU, aspectRatio, SCREEN_WIDTH);




    image->showImage("GPU");








    // Free memory and cleanup

    return 0;
}