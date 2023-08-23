import torch
from net import LSTM
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outfile", required=True, default=False)
    ap.add_argument("-m", "--model", required=True, default=False)
    ap.add_argument("--sequence", required=False, default=10,type=int)
    ap.add_argument("--layers", required=False, default=2,type=int)
    ap.add_argument("--input", required=False, default=4,type=int)
    ap.add_argument("--hidden", required=False, default=32,type=int)
    ap.add_argument("--output", required=False, default=7,type=int)
    args = ap.parse_args()

    ####################### HYPERPARAMETERS ###########################

    n_DOFs = 7                 
    input_len = args.input
    sequence_len = args.sequence
    num_layers = args.layers
    hidden_size = args.hidden
    output_size = args.output

    ###################################################################

    # Load instance of model.
    net = LSTM(num_features=n_DOFs, input_size=input_len*n_DOFs, hidden_size=hidden_size, num_layers=num_layers, seq_length=sequence_len)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    net.load_state_dict(torch.load(args.model),strict=False)


    # An example input you would normally provide to the model's forward() method.
    example = torch.rand(sequence_len,  input_len*n_DOFs).cuda()

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net.module,example)

    # save traced script module
    traced_script_module.save(args.outfile)

if __name__=='__main__':
    main()