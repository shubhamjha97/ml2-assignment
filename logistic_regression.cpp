#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string.h>

#define float  double
using namespace Eigen;
using namespace std;

// weight values.
struct params{
	Eigen :: RowVectorXd w;
	float w0;
};

// data structure for holding data features and target class
struct data_structure{
	Eigen::MatrixXd features;
	Eigen :: RowVectorXd target_class;

};

/**
    Reads data from file and stores it in the appropriate data structure.

    @param file_location: the path to the file.
    
    @return data_structure: containing the features and the target values.
*/
data_structure read_data(string file_location)
{
	ifstream trainFile;
	trainFile.open(file_location.c_str());
	vector<vector<float> > train_data;
	vector<int> target_class;
	if(trainFile.is_open())
	{
		while(!trainFile.eof())
		{
			string s;
			vector<float> temp;
			getline(trainFile, s);
			int comma_no = 0;
			int prev = 0;
			for(int i=0;i<s.length(); i++)
			{
				if(s[i]==',')
				{
					string temp_float = s.substr(prev,i-prev);
					prev = i+1;
					temp.push_back(stof(temp_float, NULL));
				}
			}
			string ss = s.substr(prev,s.length()-prev);
			int c = stoi(ss);
			target_class.push_back(c);
			train_data.push_back(temp);
		}
	}

	data_structure d;

	Eigen::MatrixXd train_data_eig(train_data.size(),train_data[0].size());
	for(int i=0;i<train_data.size();i++)
        for(int j=0;j<train_data[0].size();j++)
            train_data_eig(i,j) = train_data[i][j];

    Eigen :: RowVectorXd target_class_eig(target_class.size());
    for(int j=0;j<target_class.size();j++)
            target_class_eig(j) = target_class[j];

	d.features = train_data_eig;
	d.target_class = target_class_eig;
	return d;
}

/**
    Computes the accuracy of data provided using the prediction and actual values.

    @param predict: contains the prediction values of classes(0 or 1).
    @param actual: contains the prediction values of classes(0 or 1).
    
    @return accuracy: of the data.
*/
float accuracy(vector<int> predict,Eigen::RowVectorXd actual)
{
    float count = 0;
    for(int i=0;i<predict.size();i++)
        if(predict[i]== (int) actual(i))
            count++;
    return 100*(count/predict.size());
}

/**  
    Compute the sigmoid value of product of given parameters and given vector
    
    @param  p:  Parameter w and w0
    @param  x:  vector whose product has to be computed
    @return Sigmoid value of product of input vector and parameter w 

*/
float sigmoid(params p, Eigen::MatrixXd x)
{
    Eigen::MatrixXd m= x*p.w.transpose();
    float z =  p.w0+m(0,0);
    return 1/(exp(-z)+1);
}

/**
    Computes the prediction class(0 or 1) of a single data point provided using the threshold value given.

    @param row_vector x: which is a single data point.
    @param params: that consist of weight values.
    @param threshold: is the threshold for classification.
    
    @return Prediction class of data point (0 or 1)
*/
int prediction(Eigen::RowVectorXd x,params p)
{
    float val = sigmoid(p,x);
    return val>=0.5?1:0;
}

/**
    Train the model on the data provided using batch gradient descent.

    @param data_structure: containing the features and the target values.
    @param MAX_STEPS: max no. of steps to train for
    @param LEARNING_RATE: learning rate for gradient descent
    @return params: containing the weight parameters.
*/
params train(data_structure data, int MAX_STEPS, float LEARNING_RATE)
{
    params p;
    Eigen::RowVectorXd rr(data.features.cols());
    rr<<((double) rand() / (RAND_MAX)), ((double) rand() / (RAND_MAX)), ((double) rand() / (RAND_MAX)), ((double) rand() / (RAND_MAX));

    p.w = rr;

    p.w0=((double) rand() / (RAND_MAX));

    float cost;
    Eigen::RowVectorXd dw(data.features.cols());
    float dw0;

    for(int step=0; step<MAX_STEPS; step++)
    {
        cost=0;
        for(int j=0; j<data.features.cols(); j++)
            dw(j)=0;
        dw0=0;
        for(int i=0; i<data.features.rows(); i++)
        {
            float y = sigmoid(p, data.features.row(i));
            int t = data.target_class[i];
            cost -= (t*log(max(0.001,y))) + (1-t)*log(max(0.001,1-y));
            for(int j=0; j<data.features.cols(); j++)
                dw(j)+=(y-t)*data.features(i,j);
            dw0+=(y-t);
        }
        if(step%100==0)
            cout<<"step:"<<step<<" cost:"<<cost<<endl;
        for(int j=0; j<data.features.cols(); j++)
            {
                p.w(j)-=LEARNING_RATE*dw(j);
                p.w0-=LEARNING_RATE*dw0;
            }
    }
    return p;
}

/**
    Computes and prints precision , recall and Confusion matrix.

    @param predict: contains the prediction values of classes(0 or 1).
    @param actual: contains the prediction values of classes(0 or 1).

*/
void metrics(vector<int> predict,Eigen::RowVectorXd actual)
{
    float fp = 0,fn = 0,tp = 0,tn = 0;
    for(int i=0;i<predict.size();i++)
        if(predict[i]== (int) actual(i))
        {
            if((int)actual(i)==0)
                tn++;
            else
                tp++;
        }
        else
        {
            if((int)actual(i)==0)
                fp++;
            else
                fn++;
        }

    cout<<"Precision : " << tp/(tp+fp)<<endl;
    cout<<"Recall : " << tp/(tp+fn)<<endl;;

    cout<<"Confusion matrix"<<endl;
    int confusion_matrix[2][2];
    confusion_matrix[0][0] = tn;
    confusion_matrix[0][1] = fp;
    confusion_matrix[1][0] = fn;
    confusion_matrix[1][1] = tp;

    cout<<"TN = "<<tn<<" FP = "<<fp<<endl<<"FN = "<<fn<<"   TP = "<<tp<<endl;
}

int main()
{

	data_structure train_data;
	train_data = read_data("data/train.txt");
	params p = train(train_data,3200,0.001);
	data_structure test_data;
	test_data = read_data("data/test.txt");

	vector<int> predictions;
	for(int i = 0;i<test_data.features.rows();i++)
        predictions.push_back(prediction(test_data.features.row(i),p));
    metrics(predictions,test_data.target_class);
    cout<<"Accuracy : "<<accuracy(predictions,test_data.target_class)<<" %"<<endl;

	return 0;
}
