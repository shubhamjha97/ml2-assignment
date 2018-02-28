#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string.h>

#define float  double
using namespace Eigen;
using namespace std;

struct params{
	Eigen :: RowVectorXd w;
};

struct data_structure{
	Eigen::MatrixXd features;
	Eigen :: RowVectorXd target_class;

};


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
    w=w/length;
    p.w=w;
    return p;
}

float accuracy(vector<int> predict,Eigen::RowVectorXd actual)
{
    float count = 0;
    for(int i=0;i<predict.size();i++)
        if(predict[i]== (int) actual(i))
            count++;
    return 100*(count/predict.size());
}


float prediction(Eigen::RowVectorXd x,params p)
{
    float y = p.w*x.transpose();
    cout<<y<<endl;
    return y;//>=0.5?1:0;
}

float compute_entropy(params p,data_structure data,float fi)
{

}

float compute_threshold(params p , data_structure data)
{
    float ri;
    float min_entropy = (float)INT_MAX;
    float threshold;
    float fi;
    for(int  i = 0;i<data.features.rows()-1;i++)
    {
        fi = (prediction(data.features.row(i),p)+prediction(data.features.row(i+1),p))/2;
        if(compute_entropy(p,data,fi)<min_entropy)
        {
            threshold = fi;
            min_entropy = compute_entropy(p,data,fi);
        }

    }
    return threshold;
}

int main()
{

	data_structure train_data;
	train_data = read_data("data/train.txt");
	params p = train(train_data);//,3200,0.001);
	float threshold = compute_threshold(p,train_data);
	data_structure test_data;
	test_data = read_data("data/test.txt");

	vector<int> predictions;
	for(int i = 0;i<test_data.features.rows();i++)
        predictions.push_back(prediction(test_data.features.row(i),p));
    cout<<"Accuracy : "<<accuracy(predictions,test_data.target_class)<<" %"<<endl;

	return 0;
}




