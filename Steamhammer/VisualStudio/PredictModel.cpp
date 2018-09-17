#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <regex>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include "PredictModel.h"

using namespace std;
// ========================= Activation Function: ELUs ========================
template<typename _Tp>
int activation_function_ELUs(const _Tp* src, _Tp* dst, int length, _Tp a)
{
	if (a < 0) {
		fprintf(stderr, "a is a hyper-parameter to be tuned and a>=0 is a constraint\n");
		return -1;
	}

	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] >= (_Tp)0. ? src[i] : (a * (exp(src[i]) - (_Tp)1.));
	}

	return 0;
}

// ========================= Activation Function: Leaky_ReLUs =================
template<typename _Tp>
int activation_function_Leaky_ReLUs(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] >(_Tp)0. ? src[i] : (_Tp)0.01 * src[i];
	}

	return 0;
}

// ========================= Activation Function: ReLU =======================
template<typename _Tp>
int activation_function_ReLU(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = std::max((_Tp)0., src[i]);
	}

	return 0;
}

// ========================= Activation Function: softplus ===================
template<typename _Tp>
int activation_function_softplus(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = log((_Tp)1. + exp(src[i]));
	}

	return 0;
}

// ========================= Activation Function: softmax ===================
template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
	_Tp denom = 0.;
	for (int i = 0; i < length; ++i) {
		denom += (_Tp)(exp(src[i]));
	}
	//printf("demon:%f", demon);
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(exp(src[i]) / denom);
	}
	return 0;
}

// ========================= Activation Function: tanh ===================
template<typename _Tp>
int activation_function_tanh(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)((exp(src[i]) - exp(-src[i])) / (exp(src[i]) + exp(-src[i])));
	}

	return 0;
}


// =============================== Activation Function: sigmoid ==========================
template<typename _Tp>
int activation_function_sigmoid(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(1. / (1. + exp(-src[i])));
	}

	return 0;
}

template<typename _Tp>
int activation_function_sigmoid_fast(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(src[i] / (1. + fabs(src[i])));
	}

	return 0;
}


void print_matrix(std::vector<double> mat)
{
	fprintf(stderr, "[");
	for each (double var in mat)
	{
		fprintf(stderr, "%0.9f, ", var);
	}
	fprintf(stderr, "]\n");
}

void test_activation_function()
{
	std::vector<double> src{ 1.23f, 4.14f, -3.23f, -1.23f, 5.21f, 0.234f, -0.78f, 6.23f };
	int length = src.size();
	std::vector<double> dst(length);

	fprintf(stderr, "source vector: \n");
	print_matrix(src);
	fprintf(stderr, "calculate activation function:\n");
	fprintf(stderr, "type: sigmoid result: \n");
	activation_function_sigmoid(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: sigmoid fast result: \n");
	activation_function_sigmoid_fast(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: softplus result: \n");
	activation_function_softplus(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: ReLU result: \n");
	activation_function_ReLU(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky ReLUs result: \n");
	activation_function_Leaky_ReLUs(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky ELUs result: \n");
	activation_function_ELUs(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky tanh result: \n");
	activation_function_tanh(src.data(), dst.data(), length);
	print_matrix(dst);
	fprintf(stderr, "type: Leaky softmax result: \n");
	activation_function_softmax(src.data(), dst.data(), length);
	print_matrix(dst);
}

string& trim(string &s, char delim)
{
	if (s.empty())
	{
		return s;
	}

	s.erase(0, s.find_first_not_of(delim));
	s.erase(s.find_last_not_of(delim) + 1);
	return s;
}

//If you want to avoid reading into character arrays, 
//you can use the C++ string getline() function to read lines into strings
void ReadDataFromFileLBLIntoString()
{
	ifstream fin("data.txt");
	string s;
	while (getline(fin, s))
	{
		cout << "Read from file: " << s << endl;
	}
}

static void _split(const string &s, char delim,
	vector<string> &elems) {
	stringstream ss(s);
	string item;

	while (getline(ss, item, delim)) {
		item = trim(item, ' ');
		elems.push_back(item);
	}
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	_split(s, delim, elems);
	return elems;
}

string extract(string &values, int index, char delim) {
	if (values.length() == 0)
		return string("");

	vector<string> x = split(values, delim);
	try {
		return x.at(index);
	}
	catch (const out_of_range& e) {
		return string("");  // return empty str if out of range
	}
}

// transfer string vector to int vector
vector<int> get_iv_from_sv(const vector<string> sv)
{
	vector<int> result;
	for each (string s in sv)
	{
		result.push_back(stoi(s));
	}
	return result;
}

pair<vector<int>, vector<double>> matrix_dot(pair<vector<int>, vector<double>> m1, pair<vector<int>, vector<double>> m2)
{
	pair<vector<int>, vector<double>> null_result;
	if (m1.first.size() < 1 || m2.first.size() < 1)
	{
		cerr << "matrix dim must bigger than one!";
		return null_result;
	}
	int m1_row = m1.first[m1.first.size() - 1];
	int m2_col = m2.first[m2.first.size() - 2];
	cout << "m1_row:" << m1_row << endl;
	cout << "m2_col:" << m2_col << endl;
	int dot_num = m1_row;
	int jump_num = 1;
	int temp_num = 1; // loop times indicator
	vector<double> result_matrix_v;
	vector<int> result_size_v;

	// check whether the dot is legal
	if (m1_row != m2_col)
	{
		cerr << "the two matrix doesn't match the dot principle !";
		return null_result;
	}

	cout << "Calculate the Result..." << endl;
	// record the shape of matrix
	for (int i = 0; i < m1.first.size() - 1; i++)
	{
		result_size_v.push_back(m1.first[i]);
	}
	for (int i = 0; i < m2.first.size(); i++)
	{
		if (i != m2.first.size() - 2)
		{
			result_size_v.push_back(m2.first[i]);
		}
	}

	// jump num of m2 for each dot
	if (m2.first.size() > 2)
	{
		jump_num = m2.first[m2.first.size() - 1];
	}
	for (int i = 1; i < m2.first.size(); i++)
	{
		temp_num *= m2.first[i];
	}
	double result = 0.0;
	for (int i = 0; i < temp_num; i++)
	{
		for (int j = 0; j < m1.second.size(); j++)
		{
			result += m1.second[j] * m2.second[jump_num * j + dot_num * i];
			if (((j + 1) % dot_num) == 0)
			{
				result_matrix_v.push_back(result);
				result = 0;
			}
		}
	}
	return make_pair(result_size_v, result_matrix_v);
}

//If we were interested in preserving whitespace, 
//we could read the file in Line-By-Line using the I/O getline() function.
vector<pair<string, pair<vector<int>, vector<double>>>> load_data(char* filename)
{
	ifstream fin(filename);
	if (!fin)
	{
		cout << "Error opening " << filename << " for input" << endl;
		exit(-1);
	}
	cout << "Loading Model..." << endl;
	string s;
	vector<pair<string, pair<vector<int>, vector<double>>>> matrix_v;
	smatch m1;
	regex e1("dict\\[(.*)\\]");
	smatch m2;
	regex e2("TF.Size\\(\\[(.*)\\]\\)");
	regex elb("(.*)\\[(.*)");
	regex erb("(.*)\\](.*)");
	regex rb("\\]");
	string name;
	vector<string> size_v;
	vector<double> matrix_p;
	bool name_set = false;
	bool sizev_set = false;

	while (getline(fin, s))
	{
		if (s == "\n")
			continue;
		//cout << "Read from file: " << s << endl;
		if (regex_search(s, m1, e1))
		{
			name = m1.format("$1");
			name_set = true;
		}
		else if (regex_search(s, m2, e2))
		{
			string size = m2.format("$1");
			size_v = split(size, ',');
			sizev_set = true;
		}
		else
		{
			vector<string> data = split(s, ',');
			int data_size = data.size();
			int match_count = 0;
			for (int i = 0; i < data_size; i++)
			{
				while (regex_match(data[i], elb))
					data[i] = trim(data[i], '[');
				while (regex_match(data[i], erb))
				{
					match_count = std::distance(
						std::sregex_iterator(data[i].begin(), data[i].end(), rb),
						std::sregex_iterator());
					data[i] = trim(data[i], ']');
				}
				matrix_p.push_back(stod(data[i]));
				if (match_count == size_v.size())
				{
					vector<int> isv = get_iv_from_sv(size_v);
					reverse(isv.begin(), isv.end());
					matrix_v.push_back(make_pair(name, make_pair(isv, matrix_p)));
					matrix_p.clear();
					name_set = false;
					sizev_set = false;
				}
			}
			//cout << "param size: " << matrix_v.size() << endl;
		}
	}
	for each (pair<string, pair<vector<int>, vector<double>>> var in matrix_v)
	{

		cout << var.first << " length:" << var.second.second.size() << endl;
		cout << var.first << " size:(";
		for each (int size in var.second.first)
		{
			cout << size << ",";
		}
		cout << ")" << endl;
	}
	return matrix_v;
}


int getPredictUnit(vector<double> input_m)
{
	vector<int> input_s;
	input_s.push_back(1);
	input_s.push_back(PREDICT_FEATURE_NUM);
	pair<vector<int>, vector<double>> input = make_pair(input_s, input_m);
	vector<pair<string, pair<vector<int>, vector<double>>>> params = load_data("4c.model");
	pair<vector<int>, vector<double>> result = input;
	bool tanh_flag = false;
	bool ReLU_flag = false;
	bool softmax_flag = false;
	for each (pair<string, pair<vector<int>, vector<double>>> param in params)
	{
		regex weight_e("(.*)\\.weight");
		regex bias_e("(.*)\\.bias");
		smatch bias_match;

		// calculate a linear
		if (std::regex_match(param.first, weight_e))
		{
			pair<vector<int>, vector<double>> temp;
			temp = matrix_dot(result, param.second);
			if (temp.second.size() != 0)
			{
				result = temp;
				//cout << result.second << endl;
			}
		}
		// add bias
		else if (std::regex_search(param.first, bias_match, bias_e))
		{
			pair<vector<int>, vector<double>> temp;
			if (result.second.size() == param.second.second.size())
			{
				for (int i = 0; i < result.second.size(); i++)
				{
					result.second[i] += param.second.second[i];
				}
			}
			// check activate function
			switch (stoi(bias_match.format("$1")))
			{
			case 0:
				tanh_flag = true;
				break;
			case 2:
				ReLU_flag = true;
				break;
			case 4:
				softmax_flag = true;
				break;
			default:
				break;
			}
		}
		// activate function
		if (tanh_flag)
		{
			activation_function_tanh(result.second.data(), result.second.data(), result.second.size());
			tanh_flag = false;
		}
		if (ReLU_flag)
		{
			activation_function_ReLU(result.second.data(), result.second.data(), result.second.size());
			ReLU_flag = false;
		}
		if (softmax_flag)
		{
			activation_function_softmax(result.second.data(), result.second.data(), result.second.size());
			softmax_flag = false;
		}
	}
	// get predict unit
	vector<double>::iterator pred_unit;
	pred_unit = max_element(result.second.begin(), result.second.end());
	cout << "build_unit:" << distance(result.second.begin(), pred_unit) << endl;
	return distance(result.second.begin(), pred_unit);
}