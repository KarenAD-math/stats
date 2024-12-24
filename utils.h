#ifndef UTILS_H
#define UTILS_H

#include <QDebug>
#include <QList>
#include <QFile>
#include <QTextStream>
#include <svm.h>
#include <variant>

void printNode(svm_node* node);

//prints an observation
void printObservation(svm_node* x, int numFeatures, QTextStream& stream);

//splits data into train and test set, calls getData for casting
//std::tuple<std::vector<svm_node*>, std::vector<double>, std::vector<svm_node*>, std::vector<double>> trainTestSplit(QList<QList<double>>& XRaw, QList<double>& YRaw, double testSplit = 0.3);

//function to cast data from a QList into dynamic array
std::variant<std::vector<svm_node*>, std::vector<double>> getData(std::variant<QList<QList<double>>, QList<double>>& Data);

//function to return a prediction over the entire dataset
std::vector<double> predict(const svm_model* model, const std::vector<svm_node*>& X);

void classificationReport(const std::vector<double>& yTrue, const std::vector<double>& yPred, QTextStream& stream);

/*
std::tuple<std::vector<svm_node*>, std::vector<double>, std::vector<svm_node*>, std::vector<double>> trainTestSplit(QList<QList<double>>& XRaw, QList<double>& YRaw, double testSplit) {

    //get the size of the data
    int dataN = XRaw.size();

    //get the train size
    int trainSize = static_cast<int>((1-0.3) * dataN);

    //get the testSize
    int testSize = dataN - trainSize;

    //get the nmber of features
    int numFeatures = XRaw[0].size();

    //split without shuffling for now

    //get the X train set
    QList<QList<double>> XTrainRaw = XRaw.first(trainSize);

    //get the X test set
    QList<QList<double>> XTestRaw = XRaw.last(testSize);

    //get the Y train set
    QList<double> YTrainRaw = YRaw.first(trainSize);

    //get the Y test set
    QList<double> YTestRaw = YRaw.last(testSize);

    //get the training set
    auto [XTrain, YTrain] = getData(XTrainRaw, YTrainRaw, numFeatures, trainSize);

    //get the test set
    auto [XTest, YTest] = getData(XTestRaw, YTestRaw, numFeatures, testSize);

    //return the split data
    return std::make_tuple(XTrain, YTrain, XTest, YTest);
}
*/


/*
 * Function to read data from a csv/text file
 * Assuming data is already casted to numeric types
 * and is scaled
 * data should include target as last col of the csv
 */
std::variant<QList<QList<double>>, QList<double>> readCSV(QString &filename, QString type = "X") {

    //intialize the file instance
    QFile file(filename);

    //make a stream
    QTextStream stream(&file);

    if(file.open(QIODevice::ReadOnly))
    {
        //make sure we begin at the line just after the header
        stream.seek(1);

        if (type == "X") {

            //initialize data as 2D matrix
            QList<QList<double>> Data;

            //while we are not at the end of the file
            while (!stream.atEnd()) {

                //create new line
                QString line;

                //read the data into the line
                stream.readLineInto(&line);

                //initialize the data using the line
                QStringList lineData = line.split(",");

                //list to hold the vector of data
                QList<double> x;

                for (const QString& xPrime: lineData) {

                    //cast the data to doubles
                    //and append to the vector of features
                    x.append(xPrime.toDouble());
                }

                //append the feature data to the X list
                Data.append(x);
            }

            //close the file
            file.close();

            return Data;

        } else {

            //initialize a column vector
            QList<double> Data;

            //while we are not at the end of the file
            while (!stream.atEnd()) {

                //create new line
                QString line;

                //read the data into the line
                stream.readLineInto(&line);

                //initialize the data using the line
                double y = line.toDouble();

                //append the feature data to the X list
                Data.append(y);
            }

            //close the file
            file.close();

            return Data;
        }
    }
    else
    {
        qInfo() << "Unable to open the file!";
        exit(1);
    }
}


//function to get the data into format useb by the svm library
//returns points to the underlying storage containers
std::variant<std::vector<svm_node*>, std::vector<double>> getData(std::variant<QList<QList<double>>, QList<double>>& Data) {

    if (std::holds_alternative<QList<QList<double>>>(Data)) {

        auto XRaw = std::get<QList<QList<double>>>(Data);

        int numFeatures = XRaw[0].size();

        //initialize the vector for the X data
        std::vector<svm_node*> X;

        //iterate over all observations
        for (int i = 0; i < XRaw.size(); i++) {
            //initialize a dynamic array for this observation
            //add an extra element for the bias term
            svm_node* x = new svm_node[numFeatures + 2];

            //add the bias term
            x[0].index = 1;
            x[0].value = 1.0;

            //iterate through the features data
            for (int j = 0; j < numFeatures; j++) {

                //set the index
                x[j + 1].index = j + 2;

                //set the value of this feature
                x[j + 1].value = XRaw[i][j];

            }

            //add the terminal node
            x[numFeatures + 1].index = -1;
            x[numFeatures + 1].value = 0;

            //inspect terminal node
            //printObservation(x, numFeatures);

            //add the node to the vector
            X.push_back(x);

        }

        //return the features set
        return X;
    } else {

        //get the column vector
        auto YRaw = std::get<QList<double>>(Data);

        //initialize the vector for the Y data
        std::vector<double> Y;

        //iterate over all observations
        for (int i = 0; i < YRaw.size(); i++) {

            //append the data to the list
            Y.push_back((int)YRaw[i]);
        }

        //return the target set
        return Y;
    }
}


void printNode(svm_node node, QTextStream& stream) {

    stream << "(Feature Index: " << node.index << ", ";

    stream << "Value: " << node.value << ")\t";
}

//function to view an observation
void printObservation(svm_node* x, int numFeatures, QTextStream& stream) {

    for (int i = 0; i < numFeatures + 2; i++) {
        printNode(x[i], stream);
    }
    stream.flush();
}

//function to make predictions over a dataset
std::vector<double> predict(const svm_model* model, const std::vector<svm_node*>& X) {

    //initialize the list to hold the predictions
    std::vector<double> predictions;

    //iterate through each observation of the dataset
    for (int i = 0; i < X.size(); i++) {

        //make a prediction
        double pred = svm_predict(model, X[i]);

        //append the prediction to the list
        predictions.push_back(pred);
    }

    //return the list
    return predictions;
}

//function to compute the performance metrics
void classificationReport(const std::vector<double>& yTrue, const std::vector<double>& yPred, QTextStream& stream) {

    //initialize variables to compute
    double TP = 0.0;
    double FP = 0.0;
    double TN = 0.0;
    double FN = 0.0;

    for (int i = 0; i < yTrue.size(); i++) {
        // if predicted true
        if (yPred[i] == 1 && yTrue[i] == 1) {
            TP += 1.0; //True positive
        } else if (yPred[i] == 1 && yTrue[i] == 0) {
            FP += 1.0; //False positive
        } else if (yPred[i] == 0 && yTrue[i] == 1) {
            FN += 1.0; //False negative
        } else if (yPred[i] == 0 && yTrue[i] == 0) {
            TN += 1.0; //True negative
        }
    }

    double N = (double)yTrue.size();

    //compute accuracy
    double accuracy = (TP + TN) / N;

    //compute recall
    double precision = TP / (TP + FP);

    //compute recall
    double recall = TP / (TP + FN);

    stream.setRealNumberPrecision(2);

    stream << "Accuracy: " << accuracy << Qt::endl;
    stream << "Precision: " << precision << Qt::endl;
    stream << "Recall: " << recall << Qt::endl;

    stream.flush();
}


#endif // UTILS_H
