#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int n_features = 7;
int n_input = 21 ;
int n_layers = 2 ;
int n_hidden = 64 ;
int l_sequence = 10 ;
double p_dropout = 0.2 ;

std::tuple<torch::Tensor, torch::Tensor> hc(
            torch::zeros({n_layers, n_hidden},torch::requires_grad()), 
            torch::zeros({n_layers, n_hidden},torch::requires_grad()));

struct Net : torch::nn::Module 
{
  Net() 
  {
    // Construct and register all the submodules 
    lstm = register_module("lstm", torch::nn::LSTMCell(n_input, n_hidden));
    fc = register_module("fc", torch::nn::Linear(n_hidden, n_features));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x,hc = lstm(x, hc);
    x = torch::dropout(x,p_dropout,is_training()) ;
    x = x.contiguous().reshape({-1, n_hidden}) ;
    x = torch::relu(fc->forward(x));
    return x;
  }

  torch::nn::LSTMCell lstm{nullptr} ;
  torch::nn::Linear fc{nullptr} ;
};



