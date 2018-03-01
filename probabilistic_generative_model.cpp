#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string.h>

#define float long double
using namespace std;

using Eigen::MatrixXd;

struct params{
	vector<vector<float> >w;
	float w0;
};

struct data_structure{
	vector<vector<float> >features;
	vector<int> target_class;

};

vector<vector<float> > inverse(vector<vector<float> > mat)
{
	Eigen::MatrixXd v2(mat.size(),mat[0].size());
	for(int i=0;i<mat.size();i++)
        for(int j=0;j<mat[0].size();j++)
            v2(i,j) = mat[i][j];


	v2=v2.inverse();

	for(int i=0;i<mat.size();i++)
        for(int j=0;j<mat[0].size();j++)
             mat[i][j] = v2(i,j) ;
	return mat;
}

vector<vector<float> > mat_multiplication(vector<vector<float> > X1,vector<vector<float> > X2)
{
	vector<vector<float> > mult(X1.size(),vector<float> (X2[0].size()));
	int i, j, k;
    for (i = 0; i < X1.size(); i++)
    {
        for (j = 0; j < X2[0].size(); j++)
        {
            mult[i][j] = 0;
            for (k = 0; k < X1[0].size(); k++)
                mult[i][j] += X1[i][k]*X2[k][j];
        }
    }
    return mult;

}


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
	d.features = train_data;
	d.target_class = target_class;
	return d;
}

vector<float> vect_add(vector<float> mu,vector<float> x)
{
	if(mu.size()== x.size())
		for(int i=0; i<mu.size(); i++)
			mu[i]+=x[i];

	return mu;
}

vector<vector<float> > vect_add_2(vector<float> mu,vector<float> x)
{
	if(mu.size()== x.size())
		for(int i=0; i<mu.size(); i++)
			mu[i]+=x[i];
	vector<vector<float> > temp(mu.size(),vector<float>(1));
	for(int i=0; i<mu.size(); i++)
			temp[0][i]=mu[i];
	return temp;
}

vector<vector<float> > convert_2d(vector<float> mu)
{
	vector<vector<float> > temp(mu.size(),vector<float>(1));
	for(int i=0; i<mu.size(); i++)
			temp[i][0]=mu[i];
	return temp;
}

vector<vector<float> > mat_add(vector<vector<float> > mu,vector<vector<float> > x)
{
	if(mu.size()== x.size())
		if(mu[0].size()==x[0].size())
		for(int i=0; i<mu.size(); i++)
				mu[i] = vect_add(mu[i], x[i]);

	return mu;
}

vector<float> multiply_vec(vector<float> vec, float n)
{
	for(int i=0;i<vec.size();i++)
		vec[i]*=n;
	return vec;
}

vector<vector<float> > multiply_mat(vector<vector<float> > vec, float n)
{
	for(int i=0;i<vec.size();i++)
		vec[i] = multiply_vec(vec[i],n);

	return vec;
}

vector<vector<float> > transpose_mat(vector<vector<float> > v1)
{
	vector<vector<float> > res(v1[0].size(), vector<float> (v1.size()));
	for(int i=0; i<v1.size(); i++)
		for(int j=0; j<v1[0].size(); j++)
			res[j][i] = v1[i][j];

	return res;
}

vector <vector<float> > transpose_vec(vector<float> v1)
{
	vector<vector<float> > res(1, vector <float> (v1.size(), 0));
	for(int i=0; i<v1.size(); i++)
		res[0][i] = v1[i];
	return res;
}

void print_mat(vector<vector<float> > vec)
{
	for(int i = 0;i<vec.size();i++)
	{
		for(int j = 0;j<vec[0].size();j++)
			cout<<vec[i][j]<<" ";
		cout<<endl;
	}
}

params train(data_structure data)
{
	vector<vector<float> > X = data.features;
	vector<int> target = data.target_class;
	float N1=0, N2=0, N=X.size();
	vector <float> mu1(X[0].size(),0), mu2(X[0].size(),0);
	vector<vector<float> > s1(X[0].size(),vector<float>(X[0].size(),0)), s2(X[0].size(),vector<float>(X[0].size(),0)), cov(X[0].size(),vector<float>(X[0].size(),0));
	float w0;
	vector<float> w;

	params p;

	for(int i=0; i<N; i++)
	{
		N1+=target[i];
		N2+=1-target[i];

		if(target[i]==1)
		{
			mu1 = vect_add(mu1,X[i]);
		}
		else
		{
			mu2 = vect_add(mu2,X[i]);
		}


	}


	mu1 = multiply_vec(mu1,1.0/(float)N1);
	mu2 = multiply_vec(mu2,1.0/(float)N2);

	for(int i=0; i<N; i++)
	{

		if(target[i]==1)
		{
			s1 = mat_add(s1, mat_multiplication(convert_2d(vect_add(X[i], multiply_vec(mu1, -1))), transpose_vec(vect_add(X[i], multiply_vec(mu1, -1)))));
		}
		else
		{
			s2 = mat_add(s2, mat_multiplication(convert_2d(vect_add(X[i], multiply_vec(mu2, -1))), transpose_vec(vect_add(X[i], multiply_vec(mu2, -1)))));
		}
	}

	s1 = multiply_mat(s1, 1.0/N1);
	s2 = multiply_mat(s2, 1.0/N2);

	cov = mat_add(multiply_mat(s1,N1/(N1+N2)),multiply_mat(s2,N2/(N1+N2)));
	vector<vector<float> > temp1 = inverse(cov);
	p.w = mat_multiplication(temp1,convert_2d(vect_add(mu1,multiply_vec(mu2, -1))));
	float term2 = mat_multiplication(mat_multiplication(transpose_mat(convert_2d(mu2)),temp1),convert_2d(mu2))[0][0];
	float term1 = mat_multiplication(mat_multiplication(transpose_mat(convert_2d(mu1)),temp1),convert_2d(mu1))[0][0];
	float term3 = log(N1/N2);
    p.w0 = 0.5*(term2-term1)+term3;
	return p;
}

float sigmoid(params p,vector<float> x)
{
    float z = -p.w0 - mat_multiplication(convert_2d(x),p.w)[0][0];
    return 1/(exp(z)+1);
}

int prediction(vector<float> x,params p)
{
    float val = sigmoid(p,x);

    return val>=0.99999 ? 1 : 0;
}

float accuracy(vector<int> predict,vector<int> actual)
{
    float count = 0;
    for(int i=0;i<predict.size();i++)
        if(predict[i]==actual[i])
            count++;
    return 100*(count/predict.size());
}


void metrics(vector<int> actual,vector<int> predict)
{
    float fp = 0,fn = 0,tp = 0,tn = 0;
    for(int i=0;i<predict.size();i++)
        if(predict[i]== (int) actual[i])
        {
            if((int)actual[i]==0)
                tn++;
            else
                tp++;
        }
        else
        {
            if((int)actual[i]==0)
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
	params p = train(train_data);
	data_structure test_data;
	test_data = read_data("data/test.txt");

	vector<int> predictions;
	for(int i = 0;i<test_data.features.size();i++)
        predictions.push_back(prediction(test_data.features[i],p));
    metrics(test_data.target_class,predictions);
    cout<<"Accuracy : "<<accuracy(test_data.target_class,predictions)<<" %"<<endl;



	return 0;
}
