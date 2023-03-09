#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>
#include <random>
#include <ctime>

using namespace cv;
using namespace std;
void compute_ssd(){
    int f[8][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, 
                      {0, 2, 4, 2, 0, 0, 0, 0},
                      {0, 2, 0, 0, 0, 0, 0, 0},
                      {0, 0, 2, 0, 0, 0, 2, 0},
                      {0, 0, 0, 0, 0, 0, 2, 0},
                      {1, 2, 1, 0, 0, 2, 4, 2},
                      {0, 1, 0, 0, 0, 0, 0, 0},
                      {0, 1, 0, 0, 0, 0, 0, 0}};
    int g[3][3] = {{1,2,1},
                {0,1,0},
                {0,1,0}};
    int ssd[8][8] = {0};
    // int diff = 0;
    for (int i = 0 ; i <= 5 ; i++){
        for (int j = 0 ; j <= 5 ; j++){
            for (int m = 0 ; m < 3 ; m++){
                for (int n = 0 ; n < 3 ; n++){
                    int diff = pow((f[i + m][j + n] - g[m][n]),2);
                    ssd[i][j] += diff;
                }
            }
        }
    }
    for (int i = 0;i< 6;i++){
        for (int j = 0;j< 6;j++){
            cout << ssd[i][j] << " && ";
        }
        cout <<" \\ " << endl;
    }
    cout << endl;
}
void nrom_correlation(){
    double f[8][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, 
                      {0, 2, 4, 2, 0, 0, 0, 0},
                      {0, 2, 0, 0, 0, 0, 0, 0},
                      {0, 0, 2, 0, 0, 0, 2, 0},
                      {0, 0, 0, 0, 0, 0, 2, 0},
                      {1, 2, 1, 0, 0, 2, 4, 2},
                      {0, 1, 0, 0, 0, 0, 0, 0},
                      {0, 1, 0, 0, 0, 0, 0, 0}};
    double g[3][3] = {{1,2,1},
                {0,1,0},
                {0,1,0}};
    double cor[8][8],F[8][8] = {0};
    double sum_f,sum_g = 0;
    double G[3][3];
    for (int i = 0;i< 8;i++){
        for (int j = 0;j< 8;j++){
            sum_f += pow(f[i][j],2);
        }
    }
    for (int i = 0;i< 8;i++){
        for (int j = 0;j< 8;j++){
            F[i][j] = f[i][j]/sqrt(sum_f);
        }
    }

    for (int i = 0;i< 3;i++){
        for (int j = 0;j< 3;j++){
            sum_g += pow(g[i][j],2);
        }
    }
    for (int i = 0;i< 3;i++){
        for (int j = 0;j< 3;j++){
            G[i][j] = g[i][j]/sqrt(sum_g);
        }
    }
    for (int i = 0 ; i <= 5 ; i++){
        for (int j = 0 ; j <= 5 ; j++){
            for (int m = 0 ; m < 3 ; m++){
                for (int n = 0 ; n < 3 ; n++){      
                    cor[i][j] += F[i + m][j + n]*G[m][n];
                }
            }
        }
    }
    for (int i = 0;i< 6;i++){
        for (int j = 0;j< 6;j++){
            cout << cor[i][j] << " & ";
        }
        cout << endl;
    }
    cout << endl;
}
void compute_correlation(){
    int f[8][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, 
                      {0, 2, 4, 2, 0, 0, 0, 0},
                      {0, 2, 0, 0, 0, 0, 0, 0},
                      {0, 0, 2, 0, 0, 0, 2, 0},
                      {0, 0, 0, 0, 0, 0, 2, 0},
                      {1, 2, 1, 0, 0, 2, 4, 2},
                      {0, 1, 0, 0, 0, 0, 0, 0},
                      {0, 1, 0, 0, 0, 0, 0, 0}};
    int g[3][3] = {{1,2,1},
                {0,1,0},
                {0,1,0}};
    double cor[8][8] = {0};
    for (int i = 0 ; i <= 5 ; i++){
        for (int j = 0 ; j <= 5 ; j++){
            for (int m = 0 ; m < 3 ; m++){
                for (int n = 0 ; n < 3 ; n++){      
                    cor[i][j] += f[i + m][j + n]*g[m][n];
                }
            }
        }
    }
    for (int i = 0;i< 6;i++){
        for (int j = 0;j< 6;j++){
            cout << cor[i][j] << " & ";
        }
        cout << endl;
    }
    cout << endl;
}
int main(){
    // compute_ssd();
    // compute_correlation();
    nrom_correlation();
    return 0;
}