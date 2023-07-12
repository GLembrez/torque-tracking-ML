#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int features = 7;
int inputs = 21 ;
int layers = 2 ;
int hidden = 64 ;
int sequence = 10 ;
double dropout = 0.2 ;


// torch::Tensor h0 = torch::zeros({n_layers, n_hidden},torch::requires_grad()); 
// torch::Tensor c0 = torch::zeros({n_layers, n_hidden},torch::requires_grad());

struct LSTM_model : torch::nn::Module 
{
  torch::nn::LSTM lstm{nullptr} ;
  torch::nn::Linear fc{nullptr} ;

  LSTM_model(int n_features,int n_hidden,int n_input, int n_layers, double p_dropout) 
  {
    // Construct and register all the submodules 
    lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(n_input, n_hidden).num_layers(n_layers)));
    fc = register_module("fc", torch::nn::Linear(n_hidden, n_features));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = std::get<0>(lstm->forward(x)) ; 
    x = torch::dropout(x,dropout,is_training()) ;
    x = x.contiguous().reshape({-1, hidden}) ;
    x = torch::relu(fc->forward(x));
    return x;
  }


};

int main()
{
	LSTM_model model = LSTM_model(features,hidden,inputs,layers,dropout);
	torch::optim::Adam optimizer(
          model.parameters(), torch::optim::AdamOptions(0.0001));
	//Input
	torch::Tensor input = torch::zeros({ sequence,1, inputs });
	//Target
	torch::Tensor target = torch::zeros({ sequence,1, features });
	//Train
	for (size_t i = 0; i < 10; i++)
	{
		torch::Tensor output = model.forward(input);
		auto loss = torch::mse_loss(output.view({sequence,1, features}), target);
		std::cout << "Loss "<< i << " : " << loss.item<float>() << std::endl;
		loss.backward();
		optimizer.step();
	}
	torch::Tensor output = model.forward(input);
	std::cout << output << std::endl;
	return EXIT_SUCCESS;
}




