#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string.h>

#define float  double
using namespace Eigen;
using namespace std;

// weight values.
struct params{
	Eigen :: RowVectorXd w;
};

// data structure for holding data features and target class
struct data_structure{
	Eigen :: MatrixXd features;
	Eigen :: RowVectorXd target_class;

};

bool post,post2 = 1;// 1 for right 0 for left of threshold

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
    Train the model on the data provided.

    @param data_structure: containing the features and the target values.
    
    @return params: containing the weight parameters.
*/
params train(data_structure data)
{
    float N1 = 0,N2 = 0;
    params p;
    Eigen::RowVectorXd m1(data.features.cols()) ,m2(data.features.cols());
    m1<<0,0,0, 0;
    m2<<0,0,0, 0;
    Eigen::MatrixXd Sw(data.features.cols(),data.features.cols());
    Sw*=0;
    for(int i=0;i<data.features.rows();i++)
    {
        if(data.target_class(i)==1)
        {
            N1++;
            for(int j = 0;j<data.features.cols();j++)
                m1(j)+=data.features(i,j);
        }
        else
        {
            N2++;
            for(int j = 0;j<data.features.cols();j++)
                m2(j)+=data.features(i,j);
        }
    }

    m1 = m1*(1/N1);
    m2 = m2*(1/N2);

    Eigen::RowVectorXd temp(data.features.cols());
    temp = m2-m1;

    Eigen::MatrixXd Sb(data.features.cols(),data.features.cols());
    Sb = temp.transpose()*temp;

    for(int i=0;i<data.features.rows();i++)
    {
        if(data.target_class(i)==1)
        {
            Sw+=(data.features.row(i)-m1).transpose()*(data.features.row(i)-m1);
        }
        else
        {
             Sw+=(data.features.row(i)-m2).transpose()*(data.features.row(i)-m2);
        }
    }

    Eigen::RowVectorXd w(data.features.cols());
    w=(Sw.inverse()*(m2-m1).transpose()).transpose();
    float length=0;
    for(int i=0; i<data.features.cols(); i++)
        length+=w(i)*w(i);
    length=sqrt(length);
    w = w/length;
    p.w = w;
    return p;
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

    cout<<"Precision : " << tp/(tp+fp)<<"\n";
    cout<<"Recall : " << tp/(tp+fn)<<"\n";;

    cout<<"Confusion matrix"<<"\n";
    int confusion_matrix[2][2];
    confusion_matrix[0][0] = tn;
    confusion_matrix[0][1] = fp;
    confusion_matrix[1][0] = fn;
    confusion_matrix[1][1] = tp;

    cout<<"TN = "<<tn<<" FP = "<<fp<<"\n"<<"FN = "<<fn<<"   TP = "<<tp<<"\n";
}

/**
    Computes the prediction class(0 or 1) of a single data point provided using the threshold value given.

    @param row_vector x: which is a single data point.
    @param params: that consist of weight values.
    @param threshold: is the threshold for classification.
    
    @return Prediction class of data point (0 or 1)
*/
float prediction(Eigen::RowVectorXd x,params p,float threshold)
{
    float y = p.w*x.transpose();
    if(post2)
    {
         if(y>=threshold)
                return 1;
            return 0;
    }
    else
    {
        if(y>=threshold)
            return 0;
        return 1;
    }
}

/**
    Computes entropy by splitting data according to fi(which is the threshold).

    @param vector r: that contains value of w.transpose * x for all the data points.
    @param params: the consist of weight values.
    @param data_structure: containing the features and the target values. 
    @param fi: are the threshold values for which the entropy is calculated. 

    @return entropy: entropy value.
*/
float compute_entropy(vector<float> r, params p,data_structure data,float fi)
{
    float entropy=0;
    vector<int> positive, negative;

    for(int i=0; i<r.size(); i++)
    {
        if(r[i]<fi)
            negative.push_back(data.target_class(i));
        else
            positive.push_back(data.target_class(i));
    }

    float entropy_neg = 0;
    float entropy_pos = 0;

    float sum_neg = 0,sum_pos = 0;

    for(int i=0;i<negative.size();i++)sum_neg+=negative[i];
    for(int i=0;i<positive.size();i++)sum_pos+=positive[i];

    float p_neg = sum_neg/negative.size();
    float p_pos = sum_pos/positive.size();
    entropy_neg = -1*(p_neg*log2(max(p_neg,0.001))+(1-p_neg)*log2(max((1-p_neg),0.001)));
    entropy_pos = -1*(p_pos*log2(max(p_pos,0.001))+(1-p_pos)*log2(max((1-p_pos),0.001)));

    if(sum_neg>sum_pos)
        post = 0;
    else
        post = 1;

    return entropy_neg+entropy_pos;
}

/**
    Computes w.transpose * x for a single data point and returns it.

    @param row vector: of a single data point.
    @param params: that consist of the weight values.

    @return w.transpose * x.
*/
float compute_fischer_1d(Eigen::RowVectorXd x,params p)
{
    float y = p.w*x.transpose();
    return y;
}

/**
    Computes the threshold value for which the entropy is minimum and returns it.

    @param params: that consist of the weight values.
    @param data_structure: containing the features and the target values. 
    
    @return threshold: value for which the entropy is minimum.
*/
float compute_threshold(params p , data_structure data)
{
    float ri;
    float min_entropy = (float)INT_MAX;
    float threshold;
    float fi;
    vector<float> r;
    vector<float> r_unsorted;
    for(int  i = 0;i<data.features.rows();i++)
        r.push_back(compute_fischer_1d(data.features.row(i),p));

    r_unsorted=r;
    sort(r.begin(), r.end());


    for(int i = 0;i<data.features.rows()-1;i++)
    {
        fi = (r[i]+r[i+1])/2;
        float curr_entropy = compute_entropy(r_unsorted, p,data,fi);
        if(curr_entropy<min_entropy)
        {
            threshold = fi;
            min_entropy = curr_entropy;
            post2 = post;
        }

    }
    return threshold;
}

int main()
{

	data_structure train_data;
	train_data = read_data("data/train.txt");
	params p = train(train_data);
	float threshold = compute_threshold(p,train_data);
	cout<<"Threshold: " <<threshold<<endl;
	data_structure test_data;
	test_data = read_data("data/test.txt");

	vector<int> predictions;
	for(int i = 0;i<test_data.features.rows();i++)
       predictions.push_back(prediction(test_data.features.row(i),p,threshold));
    metrics(predictions,test_data.target_class);
    cout<<"Accuracy : "<<accuracy(predictions,test_data.target_class)<<" %"<<endl;

	return 0;
}




