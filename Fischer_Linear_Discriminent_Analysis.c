#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string.h>

#define float  double
using namespace Eigen;
using namespace std;

struct params{
	Eigen :: RowVectorXd w;
	float w0;
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

float accuracy(vector<int> predict,Eigen::RowVectorXd actual)
{
    float count = 0;
    for(int i=0;i<predict.size();i++)
        if(predict[i]== (int) actual(i))
            count++;
    return 100*(count/predict.size());
}


int prediction(Eigen::RowVectorXd x,params p)
{
    float val = sigmoid(p,x);
    return val>=0.5?1:0;
}



