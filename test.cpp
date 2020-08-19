#include "bayes.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>

using namespace std;
using namespace Eigen;

int main() {
    MatrixXf trainDataX(6,3);
    VectorXi trainDataY(6);
    trainDataX << 2.0, 2.0, 2.0,
                  0.2, 0.2, 0.2,
                  4.0, 4.0, 4.0,
                  0.4, 0.4, 0.4,
                  3.0, 3.0, 3.0,
                  0.3, 0.3, 0.3;
    trainDataY << 1, 0, 1, 0, 1, 0;
    
    auto b1 = chrono::high_resolution_clock::now();
    GNBC nbc(trainDataX, trainDataY);
    auto b2 = chrono::high_resolution_clock::now();
    cout << "Declaration and training duration = " << chrono::duration_cast<chrono::microseconds>(b2 - b1).count() << "μs" << endl;
   
    nbc.printCoefficients();
    

    MatrixXf X(50,3);
    X << 0.096730, 0.666528, 0.699888,
         0.818149, 0.178132, 0.638531,
         0.817547, 0.128014, 0.033604,
         0.722440, 0.999080, 0.068806,
         0.149865, 0.171121, 0.319600,
         0.659605, 0.032601, 0.530864,
         0.518595, 0.561200, 0.654446,
         0.972975, 0.881867, 0.407619,
         0.648991, 0.669175, 0.819981,
         0.800331, 0.190433, 0.718359,
         0.453798, 0.368917, 0.968649,
         0.432392, 0.460726, 0.531334,
         0.825314, 0.981638, 0.325146,
         0.083470, 0.156405, 0.105629,
         0.133171, 0.855523, 0.610959,
         0.173389, 0.644765, 0.778802,
         0.390938, 0.376272, 0.423453,
         0.831380, 0.190924, 0.090823,
         0.803364, 0.428253, 0.266471,
         0.060471, 0.482022, 0.153657,
         0.399258, 0.120612, 0.281005,
         0.526876, 0.589507, 0.440085,
         0.416799, 0.226188, 0.527143,
         0.656860, 0.384619, 0.457424,
         0.627973, 0.582986, 0.875372,
         0.291984, 0.251806, 0.518052,
         0.431651, 0.290441, 0.943623,
         0.015487, 0.617091, 0.637709,
         0.984064, 0.265281, 0.957694,
         0.167168, 0.824376, 0.240707,
         0.106216, 0.982663, 0.676122,
         0.372410, 0.730249, 0.289065,
         0.198118, 0.343877, 0.671808,
         0.489688, 0.584069, 0.695140,
         0.339493, 0.107769, 0.067993,
         0.951630, 0.906308, 0.254790,
         0.920332, 0.879654, 0.224040,
         0.052677, 0.817761, 0.667833,
         0.737858, 0.260728, 0.844392,
         0.269119, 0.594356, 0.344462,
         0.422836, 0.022513, 0.780520,
         0.547871, 0.425259, 0.675332,
         0.942737, 0.312719, 0.006715,
         0.417744, 0.161485, 0.602170,
         0.983052, 0.178766, 0.386771,
         0.301455, 0.422886, 0.915991,
         0.701099, 0.094229, 0.001151,
         0.666339, 0.598524, 0.462449,
         0.539126, 0.470924, 0.424349,
         0.698106, 0.695949, 0.460916;

    cout << "Dimensions of input matrix = " << X.rows() << "x" << X.cols() << endl;
    
    auto k1 = chrono::high_resolution_clock::now();
    MatrixXf predictions = nbc.predictMatrix(X);
    auto k2 = chrono::high_resolution_clock::now();
    cout << "Prediction =\n" << predictions << endl << "Prediction duration = " << chrono::duration_cast<chrono::microseconds>(k2 - k1).count() << "μs" << endl;
    
    auto a1 = chrono::high_resolution_clock::now();
    VectorXi labels = nbc.labelMatrix(predictions);
    auto a2 = chrono::high_resolution_clock::now();
    cout << "Labeling =\n" << labels << endl << "Labeling duration = " << chrono::duration_cast<chrono::microseconds>(a2 - a1).count() << "μs" << endl;
    
    return 0;
}
