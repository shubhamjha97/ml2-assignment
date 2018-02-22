#include <bits/stdc++.h>

using namespace std;


struct params{
	vector<float> w;
	float w0;
};

struct data_structure{
	vector<vector<float> >features;
	vector<int> target_class;

};

vector<vector<float> > mat_multiplication(vector<vector<float> > X1,vector<vector<float> > X2)
{
	vector<vector<float> > mult(X1.size(),vector<float> (X2[0].size()));
	// Multiplying matrix a and b and storing in array mult.
    for(int i = 0; i < X1.size(); ++i)
        for(int j = 0; j < X2[0].size(); ++j)
            for(int k = 0; k < X1[0].size(); ++k)
            {
                mult[i][j] += X1[i][k] * X2[k][j];
            }

    return mult;

}


data_structure read_data(string file_location)
{
	ifstream trainFile;
	trainFile.open(file_location);
	vector<vector<float> > train_data;
	vector<int> target_class;
	if(trainFile.is_open())
	{
		while(!trainFile.eof())
		{
			string s;
			vector<float> temp;
			getline(trainFile, s);//, ',');
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

	data_structure d;// = (data_structure)malloc(sizeof(data_structure));
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

vector<vector<float> > vect_add(vector<vector<float> > mu,vector<vector<float> > x)
{
	if(mu.size()== x.size())
		if(mu[0].size()==x[0].size())
		for(int i=0; i<mu.size(); i++)
				mu[i] = vect_add(mu[i], x[i]);

	return mu;
}

vector<float> mulltiply_vec(vector<float> vec,float n)
{
	for(int i=0;i<vec.size();i++)
		vec[i]*=n;
	return vec;
} 

vector<vector<float> > multiply_vec(vector<vector<float> > vec,float n)
{
	for(int i=0;i<vec.size();i++)
		vec[i] = opertaion_vec(vec[i],n);
		
	return vec;
}

vector<vector<float> > transpose(vector<vector<float> > v1):
{
	vector<vector<float> > res(v1[0].size(), vector<float> (v1.size()));
	for(int i=0; i<v1.size(); i++)
		for(int j=0; j<v1[0].size(); j++)
			res[j][i] = v1[i][j];

	return res;
}


params train(data_structure data)
{
	vector<vector<float> > X = data.features;
	vector<int> target = data.target_class;
	int N1=0, N2=0, N=X.size();
	vector <float> mu1, mu2;
	vector<vector<float> > s1, s2, cov;
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

	mu1 = multiply_vec(mu1,1/N1);
	mu2 = multiply_vec(mu2,1/N2);

	for(int i=0; i<N; i++)
	{

		if(target[i]==1)
		{

			s1 = mat_multiplication()
		}
		else
		{
			
		}

	}


	return p;
}


int main()
{

	data_structure train_data;// = (data_structure)malloc(sizeof(data_structure));
	train_data = read_data("data/train.txt");
	params p = train(train_data);
	//cout<<train_data.features[0][0];
	
	return 0;
}