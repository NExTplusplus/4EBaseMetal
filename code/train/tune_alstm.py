import argparse
from grid_search import grid_search_alstm


if __name__ == '__main__':
    desc = 'Tune hyperparameters of alstm (bilstm + attention)'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--log', type=str, default='./tune.log',
                        help='path and name of log file')
    parser.add_argument('--script', type=str, default='code/train/train_alstm.py',
                        help='name of the script to train model')
    parser.add_argument('--drop_out', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--hidden', type=int, default=50,
                        help='number of units in hidden layer')
    parser.add_argument('--embedding_size', type=int, default=5,
                        help='number of units in metal embedding')
    parser.add_argument('--lag', type=int, default=3,
                        help='window size for features')
    parser.add_argument('--batch', type=int, default=512,
                        help='batch size')
    parser.add_argument('--steps', type=int, default=5,
                        help='time horizon')

    args = parser.parse_args()
    print(args)

    selected_parameters = ['lag', 'hidden', 'embedding_size', 'drop_out', 'batch']
    # parameter_values = [
    #     [3, 5, 10, 20],
    #     [30, 50, 100],
    #     [5, 10, 20],
    #     [0.0, 0.2, 0.4, 0.6],
    #     [512, 1024]
    # ]
    # parameter_values = [
    #     [2, 3, 4, 5],
    #     [30, 50, 70],
    #     [20, 30],
    #     [0.0, 0.2, 0.4, 0.6],
    #     [256, 512]
    # ]
    parameter_values = [
        [2, 3, 4, 5],
        [10, 20, 30],
        [5, 10, 20],
        [0.2, 0.4, 0.6],
        [512]
    ]
    init_para = {
        'drop_out': args.drop_out,
        'hidden': args.hidden,
        'embedding_size': args.embedding_size,
        'batch': args.batch,
        'lag': args.lag
    }

    grid_search_alstm(selected_parameters, parameter_values, init_para,
                script=args.script, log_file=args.log, steps=args.steps)
