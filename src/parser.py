import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Twitch Brasilians dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run BANE.")


    parser.add_argument('--edge-path',
                        nargs = '?',
                        default = './input/ptbr_edges.csv',
	                help = 'Edge list csv.')

    parser.add_argument('--output-path',
                        nargs = '?',
                        default = './output/ptbr_danmf.csv',
	                help = 'Target embedding csv.')

    parser.add_argument('--membership-path',
                        nargs = '?',
                        default = './output/ptbr_membership.json',
	                help = 'Cluster membership json.')

    parser.add_argument('--pre-training-method',
                        nargs = '?',
                        default = 'shallow',
	                help = 'Pre-training procedure used.')

    parser.add_argument('--iterations',
                        type = int,
                        default = 10,
	                help = 'Number of power iterations. Default is 10.')

    parser.add_argument('--pre-iterations',
                        type = int,
                        default = 100,
	                help = 'Number of CDC rounds. Default is 5.')

    parser.add_argument('--seed',
                        type = int,
                        default = 42,
	                help = 'Random seed for sklearn pre-training. Default is 42.')

    parser.add_argument('--lamb',
                        type = float,
                        default = 0.01,
	                help = 'Trade-off parameter. Default is 0.7.')

    parser.add_argument('--layers',
                        nargs='+',
                        type=int,
                        help = 'Layer dimensions separated by space. E.g. 128 64 32.')

    parser.add_argument('--calculate-loss',
                        dest='calculate_loss',
                        action='store_true')

    parser.add_argument('--not-calculate-loss',
                        dest='calculate_loss',
                        action='store_false')

    parser.set_defaults(calculate_loss=False)
    
    return parser.parse_args()

