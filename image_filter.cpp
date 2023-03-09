#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>
#include <random>
#include <ctime>

using namespace cv;
using namespace std;

void get_img_value(Mat image){
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cout << "(";
            for (int k = 0; k < image.channels(); k++) {
                cout << (int)image.at<Vec3b>(i, j)[k] << ",";
            }
            cout << ") ";
        }
        cout << endl;
    }
}
void noisy_img_gen(){
    // This algorithm generates a noisy image with zero mean and standard deviation of 2.0
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 2.0);

    Mat base_image(256, 256, CV_8UC1, Scalar(128));
    generator.seed(time(0));
    for (int i = 0; i < 10; i++) {
      Mat noisy_image = base_image.clone();
      for (int r = 0; r < 256; r++) {
        for (int c = 0; c < 256; c++) {
          double noise = (double)distribution(generator);
          noisy_image.at<uchar>(r, c) = saturate_cast<uchar>(noisy_image.at<uchar>(r, c) + noise);
        //   cout << noisy_image.at<uchar>(r, c) << endl;
        }
      }
      String filename = "noisy_image_" + to_string(i) + ".jpg";
      imwrite(filename, noisy_image);
    }
}
double EST_NOISE(String image_name){
    double bgrPixel[256][256] = {0};
    double sigma[256][256] = {0};

    for(uint8_t s = 0;s<10;s++){
        string file_name = (String)image_name + to_string(s) + ".jpg";
        Mat image = imread(file_name, IMREAD_COLOR);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                bgrPixel[i][j] += (sum(image.at<Vec3b>(i, j))/3)[0];
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            bgrPixel[i][j] /= 10;
            // cout << bgrPixel[i][j] << ",";
        }
    }
    for(uint8_t s = 0;s<10;s++){
        string file_name = (String)image_name + to_string(s) + ".jpg";
        Mat image = imread(file_name, IMREAD_COLOR);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                double diff = (bgrPixel[i][j] - (sum(image.at<Vec3b>(i, j))/3)[0]);
                sigma[i][j] += pow(diff,2);
                // cout << sigma[i][j] << ",";
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            sigma[i][j] /= 9;
            sigma[i][j] = pow(sigma[i][j],0.5);
        }
    }
    double sig_est = 0.0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            sig_est += sigma[i][j];
            // cout << bgrPixel.at<Vec3f>(i, j) << endl;
        }
    }
    return sig_est/65536;
}
void box_filter(){
    Mat kernel = Mat::ones(1, 3, CV_32F) / 9;
    Mat filteredImage;
    for(uint8_t s = 0;s<10;s++){
        string file_name = "noisy_image_" + to_string(s) + ".jpg";
        Mat image = imread(file_name);
        filter2D(image, filteredImage, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
        imwrite("filtered_image.jpg", filteredImage);
        for(uint8_t i = 0;i<10;i++){
            string file_name = "filter_image_" + to_string(i) + ".jpg";
            imwrite(file_name,image);    
        }
    }
}
void line_filter(){
    Mat kernel1 = Mat::ones(1, 5, CV_32F) / 5;

    float B[1][5] = {0.1,0.2,0.4,0.2,0.1};
    Mat kernel2(1,5,CV_32F,B);

    uint8_t noisy_img[1][10] = {10,10,10,10,10,40,40,40,40,40};
    Mat filt_img = Mat::zeros(1, 10, CV_32F);
    Mat A(1,10,CV_8U,noisy_img);
    clock_t strt = clock();
    // filter2D(A,filt_img,-1,kernel2, Point(-1,-1),0,BORDER_CONSTANT);
    filter2D(A,filt_img,-1,kernel1, Point(-1,-1),0,BORDER_CONSTANT);
    clock_t fin = clock();
    double elapsed = 1000.0*(fin-strt) / CLOCKS_PER_SEC;
    double fil_mean = mean(filt_img)[0];
    double var = 0.0;
    cout << filt_img << endl;
    Mat diff = fil_mean - filt_img;
    cout << diff << endl;
    cout << filt_img << endl;
    // Scalar mean = mean(filt_img);

}
void gen_gaussian_mask(){
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.4);
    double mask[8][8] = {0};
    for (int i = 0; i < 8; i++) {
        for (int r = 0; r < 8; r++) {
            double noise = (double)distribution(generator);
            mask[i][r] = noise;
            cout << mask[i][r] << endl;
      }
    }
}
void prewitt_mask(){
    Mat prewitt_x = (Mat_<float>(3,3) << -1,0,1,-1,0,1,-1,0,1);
    Mat prewitt_y = (Mat_<float>(3,3) << 1,1,1,0,0,0,-1,-1,-1);
    Mat gradient_x;
    Mat gradient_y;
    Mat img = Mat::zeros(8,8,CV_32F);
    for (uint8_t i = 0; i < 9; i++) {
        for (uint8_t r = 0; r < 9; r++) {
            img.at<float>(i,r) = abs(i - r);
            // cout << img.at<float>(i,r) << endl;
      }
    }
    filter2D(img,gradient_x,CV_32F,prewitt_x);
    filter2D(img,gradient_y,CV_32F,prewitt_y);
    Mat magnitude = Mat::zeros(img.rows, img.cols, CV_32F);
    Mat orientation = Mat::zeros(img.rows, img.cols, CV_32F);
    for (int i = 0; i < img.rows; i++) {
      for (int j = 0; j < img.cols; j++) {
        float gx = gradient_x.at<float>(i, j);
        float gy = gradient_y.at<float>(i, j);
        magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);
        orientation.at<float>(i, j) = atan2(gy, gx);
      }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
          cout << orientation.at<float>(i, j) << " ";
        }
        cout << endl;
  }

}
void soble_masks(){
    Mat soble_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat soble_y = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
    Mat gradient_x;
    Mat gradient_y;
    Mat img = Mat::zeros(8,8,CV_32F);
    for (uint8_t i = 0; i < 9; i++) {
        for (uint8_t r = 0; r < 9; r++) {
            img.at<float>(i,r) = abs(i - r);
            // cout << img.at<float>(i,r) << endl;
      }
    }
    filter2D(img,gradient_x,CV_32F,soble_x);
    filter2D(img,gradient_y,CV_32F,soble_y);
    Mat magnitude = Mat::zeros(img.rows, img.cols, CV_32F);
    Mat orientation = Mat::zeros(img.rows, img.cols, CV_32F);
    for (int i = 0; i < img.rows; i++) {
      for (int j = 0; j < img.cols; j++) {
        float gx = gradient_x.at<float>(i, j);
        float gy = gradient_y.at<float>(i, j);
        magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);
        orientation.at<float>(i, j) = atan2(gy, gx);
      }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
          cout << magnitude.at<float>(i, j) << " ";
        }
        cout << endl;
    }
    cout << "oreintation" << endl;
    for (int i = 0; i < img.rows; i++) {
          for (int j = 0; j < img.cols; j++) {
            cout << orientation.at<float>(i, j) << " ";
          }
          cout << endl;
    }
}
 
void corner_detecy(){

    Mat Zeroes = Mat::zeros(10,10,CV_32F);
    Mat Onees = 40*Mat::ones(10,10,CV_32F);
    Mat img;
    vconcat(Zeroes, Onees, img);
    Mat img3;
    vconcat(Onees,Zeroes, img3);
    Mat img2;
    hconcat(img, img3, img2);
    Mat Ix = Mat::zeros(20,20,CV_32F);
    Mat Iy = Mat::zeros(20,20,CV_32F);
    for (int i = 0; i < img2.rows; i++) {
          for (int j = 0; j < img2.cols; j++) {
            cout << img2.at<float>(i, j) << " ";
          }
          cout << endl;
    }
    Mat prewitt_x = (Mat_<float>(3,3) << -1,0,1,-1,0,1,-1,0,1);
    Mat prewitt_y = (Mat_<float>(3,3) << 1,1,1,0,0,0,-1,-1,-1);
    Mat gradient_x;
    Mat gradient_y;
    filter2D(img2,gradient_x,CV_32F,prewitt_x);
    filter2D(img2,gradient_y,CV_32F,prewitt_y);
    Mat C = Mat::zeros(img2.rows, img2.cols, CV_32F);
    for (int i = 0; i < img2.rows; i++) {
      for (int j = 0; j < img2.cols; j++) {
        float gx = gradient_x.at<float>(i, j);
        float gy = gradient_y.at<float>(i, j);
        C.at<float>(i, j) = gx * gx + gy * gy;
        // cout << C.at<Vec2f>(i, j) << " ";
      }
    }
    Mat eigenvalues = Mat::zeros(20,20,CV_32FC2);
    eigen(C,eigenvalues);
}
int main()
{
    // prewitt_mask();
    // soble_masks();
    corner_detecy();

    // int width = 256;
    // int height = 256;
    // const double mean = 0.0;
    // const double stde = 2.0;
    // Mat img(width,height,CV_8UC1,Scalar(128));
    // Mat noise(img.size(),img.type());
    // randn(noise,mean,stde);
    // imwrite("test.jpg",img);
    // img += noise;
    // // int blue = result[1];
    // printf("%lf",result);
    // get_img_value(img);
    // if (img.empty()){
    //     printf("No image was created");
    // }
    // for(uint8_t i = 0;i<10;i++){
    //     string file_name = "raw_image_" + to_string(i) + ".jpg";
    //     imwrite(file_name,img);    
    // }
    // noisy_img_gen();
    // double est_noisy = EST_NOISE("noisy_image_");
    // box_filter();
    // double est_filtered = EST_NOISE("filter_image_");
    // printf("noisy image noise %f and filtered image noise %f",est_noisy,est_filtered);
    // line_filter();
    // gen_corner_img();
    // gen_gaussian_mask();
    // box_filter();
    // for(uint8_t s = 0;s<10;s++){
    //     string file_name = "image_" + to_string(s) + ".jpg";  
    //     Mat image = imread(file_name);
    //     for (int i = 0; i < image.rows; i++) {
    //         for (int j = 0; j < image.cols; j++) {
    //             for (int k = 0; k < image.channels(); k++) {
    //                 cout << (double)image.at<Vec3b>(i, j)[k] << endl;
    //             }
    //         }
    //     }
    // }
    // // namedWindow("Temp",cv::WINDOW_AUTOSIZE);
    // // imshow("Temp",img);
    // waitKey(0);
    return 0;   
}