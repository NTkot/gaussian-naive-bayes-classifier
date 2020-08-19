#include "bayes.h"
#include <iostream>

#ifndef PI
#define PI 3.14159265
#endif

GNBC::GNBC() {
    numberOfClasses = 0;
    numberOfFeatures = 0;
}

GNBC::GNBC(Eigen::MatrixXf &trainDataX, Eigen::VectorXi &trainDataY) {
    train(trainDataX, trainDataY);
}

void GNBC::train(Eigen::MatrixXf &trainDataX, Eigen::VectorXi &trainDataY) {
    numberOfFeatures = trainDataX.cols();
    numberOfClasses = trainDataY.maxCoeff() + 1;
    means = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(numberOfClasses, numberOfFeatures);
    variances = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(numberOfClasses, numberOfFeatures);
    numberOfExamples = Eigen::VectorXi::Zero(numberOfClasses);
    classesPropabilities = Eigen::VectorXf::Zero(numberOfClasses);
    if(trainDataX.rows() != trainDataY.size()) {
        std::cerr << "TrainDataX and TrainDataY incompatible dimensions\n";
        return;
    }
    for(int i = 0; i < trainDataX.rows(); i++) {
        numberOfExamples(trainDataY(i))++;
        Eigen::VectorXf temp = means.row(trainDataY(i)) + (trainDataX.row(i)-means.row(trainDataY(i))) / numberOfExamples(trainDataY(i));
        variances.row(trainDataY(i)) = variances.row(trainDataY(i)) + (((float)numberOfExamples(trainDataY(i))-1) / (float)numberOfExamples(trainDataY(i)) * (trainDataX.row(i)-means.row(trainDataY(i))).cwiseProduct(trainDataX.row(i)-means.row(trainDataY(i))) - variances.row(trainDataY(i))) / (float)numberOfExamples(trainDataY(i));
        means.row(trainDataY(i)).noalias() = temp;
    }
    variances = (variances.array().colwise() * (numberOfExamples.cast<float>().array() / (numberOfExamples.cast<float>().array() - 1))).matrix();
    classesPropabilities = numberOfExamples.cast<float>() / (numberOfExamples.sum());
}

Eigen::VectorXf GNBC::predict(const Eigen::VectorXf &X) {
    Eigen::VectorXf probs = Eigen::VectorXf::Zero(numberOfClasses);
    float Z = classesPropabilities.cwiseProduct(gauss(X)).sum();
    probs = 1/Z*classesPropabilities.cwiseProduct(gauss(X));
    return probs;
}

Eigen::MatrixXf GNBC::predictMatrix(const Eigen::MatrixXf &X) {
    Eigen::MatrixXf probs = Eigen::MatrixXf::Zero(X.rows(), numberOfClasses);
    float Z;
    for(int i = 0; i < X.rows(); i++) {
        Z = classesPropabilities.cwiseProduct(gauss(X.row(i).transpose())).sum();
        probs.row(i) = predict(X.row(i).transpose()).transpose();
    }
    return probs;
}

Eigen::VectorXf GNBC::gauss(const Eigen::VectorXf &X) {
    Eigen::VectorXf g = Eigen::VectorXf::Zero(numberOfClasses);
    for(int i = 0; i < numberOfClasses; i++)
        g(i) = (2*PI*variances.row(i)).cwiseSqrt().cwiseInverse().cwiseProduct((((X.transpose()-means.row(i)).cwiseProduct(X.transpose()-means.row(i))).cwiseQuotient(-2*variances.row(i))).array().exp().matrix()).prod();
    return g;
}

int GNBC::label(const Eigen::VectorXf &prediction) {
    int index;
    prediction.maxCoeff(&index);
    return index;
}

Eigen::VectorXi GNBC::labelMatrix(const Eigen::MatrixXf &predictionMatrix) {
    Eigen::VectorXi indexes(predictionMatrix.rows());
    for(int i = 0; i < predictionMatrix.rows(); i++)
        indexes(i) = label(predictionMatrix.row(i).transpose());
    return indexes;
}

void GNBC::printCoefficients() {
    std::cout << "means =\n" << means << std::endl << std::endl
              << "variances =\n" << variances << std::endl << std::endl
              << "numberOfExamples =\n" << numberOfExamples << std::endl << std::endl
              << "classesPropabilites =\n" << classesPropabilities << std::endl << std::endl;
}

void GNBC::printSizes() {
    std::cout << "numberOfClasses = " << numberOfClasses << std::endl
              << "numberOfFeatures = " << numberOfFeatures << std::endl
              << "numberOfExamples = " << numberOfExamples << std::endl;
}
