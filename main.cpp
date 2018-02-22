#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;


struct params{
	vector<float> w;
	float w0;
};

struct data_structure{
	vector<vector<float> >features;
	vector<int> target_class;

};

void getCofactor(int A[N][N], int temp[N][N], int p, int q, int n)
{
    int i = 0, j = 0;
 
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];
 
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
 
/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][]. */
int determinant(vector<vector<float> > A)//, int n)
{
    float D = 0; // Initialize result
 	
    //  Base case : if matrix contains single element
    if (n == 1)
        return A[0][0];
 
    float temp[A.size()][N]; // To store cofactors
 
    float sign = 1;  // To store sign multiplier
 
     // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);
 
        // terms are to be added with alternate sign
        sign = -sign;
    }
 
    return D;
}
 
// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(int A[N][N],int adj[N][N])
{
    if (N == 1)
    {
        adj[0][0] = 1;
        return;
    }
 
    // temp is used to store cofactors of A[][]
    int sign = 1, temp[N][N];
 
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            // Get cofactor of A[i][j]
            getCofactor(A, temp, i, j, N);
 
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;
 
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j][i] = (sign)*(determinant(temp, N-1));
        }
    }
}
 
// Function to calculate and store inverse, returns false if
// matrix is singular
vector<vector<float> > inverse(vector<vector<float> > A)//, <float> inverse[N][N])
{
    // Find determinant of A[][]
    int det = determinant(A);
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return NULL;
    }
 
    // Find adjoint
    float adj[N][N];
    adjoint(A, adj);
 
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    vector<vector<float> > inverse(A[0].size(),vector<float> (A.size()));
    for (int i=0; i<A[0].size(); i++)
        for (int j=0; j<A.size(); j++)
            inverse[i][j] = adj[i][j]/float(det);
 
    return inverse;
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
            //cout<<i<<" "<<j<<endl;
            for (k = 0; k < X1[0].size(); k++)
                mult[i][j] += X1[i][k]*X2[k][j];
        }
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
	//cout<<"asd";
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
	//cout<<"Calling Function"<<endl<<endl;
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

	//for(int i=0;i<X[0].size();i++)cout<<X[0][i]<<" ";
	//cout<<X[0].size();
		//print_mat(convert_2d(X[0]));
	for(int i=0; i<N; i++)
	{

		if(target[i]==1)
		{
			

			//print_mat(mat_multiplication(convert_2d(X[0]),transpose_vec(X[0])));
			s1 = mat_add(s1, mat_multiplication(convert_2d(vect_add(X[i], multiply_vec(mu1, -1))), transpose_vec(vect_add(X[i], multiply_vec(mu1, -1)))));
		
			//print_mat(s1);
			//cout<<endl<<endl;
		}
		else
		{
			s2 = mat_add(s2, mat_multiplication(convert_2d(vect_add(X[i], multiply_vec(mu2, -1))), transpose_vec(vect_add(X[i], multiply_vec(mu2, -1)))));
		}
	}

	s1 = multiply_mat(s1, 1.0/N1);
	s2 = multiply_mat(s2, 1.0/N2);
	//print_mat(s1);
	//cout<<endl<<N1<<" "<<N2<<endl<<endl;
	cov = mat_add(multiply_mat(s1,N1/(N1+N2)),multiply_mat(s2,N2/(N1+N2)));
	//print_mat(cov);

	return p;
}


int main()
{

	data_structure train_data;// = (data_structure)malloc(sizeof(data_structure));
	train_data = read_data("data/train.txt");
	params p = train(train_data);
	Matr
	//cout<<train_data.features[0][0];
	
	return 0;
}