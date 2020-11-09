import os
import sys
import argparse
from copy import copy
sys.path.insert(0,os.path.abspath(os.path.join(sys.path[0],"..")))
from live.Logistic_live import Logistic_online

if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)

    #script parameters
    parser.add_argument('-o', '--action', type=str,
                        help='action that we wish to take, has potential values of : train, test, tune',
                        default='train'
                        )

    #result parameters
    parser.add_argument('-s','--horizon',type=int,
                        help='the prediction horizon',
                        default=5
                        )
    parser.add_argument('-gt', '--ground_truth', type=str, 
                        help='the name of the column that we are predicting either value or direction',
                        default="LME_Co_Spot"
                        )
    parser.add_argument('-sou','--source', type = str, 
                        help='source of data', 
                        default = "NExT"
                        )
    parser.add_argument('-v','--version', type = str, 
                        help='feature version for data', 
                        default = 'v10'
                        )
    parser.add_argument('-d', '--date', type=str,
                        help = "string of comma-separated dates which identify the total period of deployment by half-years"
                        )

    #hyperparameters
    parser.add_argument('-l','--lag', type=int, 
                        help='lag',
                        default = 5
                        )
    parser.add_argument('-max_iter','--max_iter',type=int,
                        help='max number of iterations',
                        default=100
                        )
    parser.add_argument('-C', '--C', type=float,
                        help = 'inverse of learning rate'
                        )
    args = parser.parse_args()

    #initialize model
    model = Logistic_online(lag = args.lag, horizon = args.horizon, version = args.version, gt = args.ground_truth, date = args.date, source = args.source)
    
    #case if action is tune
    if args.action=="tune":
        model.tune(100)

    #case if action is train
    elif args.action=='train':
        model.train(C=args.C, max_iter=args.max_iter)
    
    #case if action is test
    else:
        final = model.test()
