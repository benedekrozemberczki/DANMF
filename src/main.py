from danmf import DANMF
from parser import parameter_parser
from utils import read_graph, tab_printer, loss_printer

def main():
    """
    Parsing command lines, creating target matrix, fitting DANMF and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    graph = read_graph(args)
    model = DANMF(graph, args)
    model.pre_training()
    model.training()
    if args.calculate_loss: 
        loss_printer(model.loss)


if __name__ =="__main__":
    main()

