#include <QCoreApplication>
#include <QFile>
#include <QDir>
#include <QTextStream>
#include "utils.h"
#include <QSharedPointer>
#include "svm2.h"
#include <iostream>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    //stream for printing
    QTextStream ostream(stdout);

    //initialize data paths
    QString XTrainPath = "data/x_train.csv";
    QString XTestPath = "data/x_test.csv";
    QString YTrainPath = "data/y_train.csv";
    QString YTestPath = "data/y_test.csv";

    qInfo() << "Reading and preparing data." << Qt::endl;

    //get the data form the csv
    auto XTrainRawVar = readCSV(XTrainPath);
    auto XTestRawVar = readCSV(XTestPath);
    auto YTrainRawVar = readCSV(YTrainPath, "Y");
    auto YTestRawVar = readCSV(YTestPath, "Y");

    //get the data in libsvm format
    auto XTrain = std::get<std::vector<svm_node*>>(getData(XTrainRawVar));
    auto XTest =  std::get<std::vector<svm_node*>>(getData(XTestRawVar));
    auto YTrain = std::get<std::vector<double>>(getData(YTrainRawVar));
    auto YTest = std::get<std::vector<double>>(getData(YTrainRawVar));

    qInfo() << "Setting up model problem and parameters." << Qt::endl;

    //set up the SVM problem
    svm_problem prob;

    //number of training samples
    prob.l = XTrain.size();

    //labels for each sample
    prob.y = YTrain.data();

    //set the data
    prob.x = XTrain.data();

    //set up the SVM parameters
    svm_parameter params;

    params.svm_type = C_SVC;
    params.kernel_type = RBF;
    params.gamma = 0.5;
    params.C = 0.5; //regularization parameter
    params.eps = 1e-3; //stopping criteria
    params.shrinking = 1;
    params.cache_size = 200;
    params.probability = 0;
    params.nr_weight = 2;

    //set the weight labels
    int* weight_label = new int[2]; //array to store the weight label for the classes
    double* weight = new double[2];

    weight_label[0] = 0;
    weight_label[1] = 1;

    weight[0] = 1/0.45;
    weight[1] = 1/0.55;

    params.weight_label = weight_label;
    params.weight = weight;

    qInfo() << "Checking parameters." << Qt::endl;

    //check the parameters for validity
    const char *errorMsg = svm_check_parameter(&prob, &params);

    if (errorMsg) {
        std::cerr << "Error: " << errorMsg << std::endl;
        return 1;
    }

    /*

    qInfo() << "Performing cross-validation." << Qt::endl;

    // Number of folds for cross-validation
    int numFolds = 5;

    // Array to store cross-validation results
    std::vector<double> target(prob.l);

    // Perform cross-validation
    svm_cross_validation(&prob, &params, numFolds, target.data());

    // Calculate the average accuracy from cross-validation results
    double correct = 0;
    for (int i = 0; i < prob.l; ++i) {
        if (target[i] == YTrain[i]) {
            ++correct;
        }
    }
    double crossValidationAccuracy = correct / prob.l * 100.0;

    qInfo() << "Cross-Validation Accuracy:" << crossValidationAccuracy << "%";

    */


    qInfo() << "Training model." << Qt::endl;

    //train the model with the given params
    svm_model *model = svm_train(&prob, &params);

    if (model == nullptr) {
        std::cerr << "Error: Failed to train the SVM model." << std::endl;
        return 1;
    }

    qInfo() << "Saving model." << Qt::endl;

    int success = svm_save_model("models/model2.txt", model);

    //qInfo() << "Loading model." << Qt::endl;

    //svm_model* model = svm_load_model("models/model1.txt");

    qInfo() << "Making Prediction." << Qt::endl;

    //make predictions on the set
    std::vector<double> pred = predict(model, XTest);

    qInfo() << "Computing performance metrics." << Qt::endl;

    //print the classification report
    classificationReport(YTest, pred, ostream);

    return a.exec();
}
