#ifndef GAUSSIAN_NAIVE_BAYES
#define GAUSSIAN_NAIVE_BAYES

#include <eigen3/Eigen/Dense>

class GNBC {
    private:
        int numberOfClasses;
        int numberOfFeatures;
        Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> means;
        Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> variances;
        Eigen::VectorXf classesPropabilities;
        Eigen::VectorXi numberOfExamples;

    public:
        GNBC();
        GNBC(int numberOfClasses, int numberOfFeatures);
        GNBC(Eigen::MatrixXf &trainDataX, Eigen::VectorXi &trainDataY);
        
        void train(Eigen::MatrixXf &trainDataX, Eigen::VectorXi &trainDataY);
        Eigen::VectorXf predict(const Eigen::VectorXf &X);
        Eigen::MatrixXf predictMatrix(const Eigen::MatrixXf &X);
        Eigen::VectorXf gauss(const Eigen::VectorXf &X);
        int label(const Eigen::VectorXf &prediction);
        Eigen::VectorXi labelMatrix(const Eigen::MatrixXf &predictionMatrix);
        void printCoefficients();
        void printSizes();
};

#endif // GAUSSIAN_NAIVE_BAYES_INCLUDED
