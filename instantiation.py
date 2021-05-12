from torch.optim import Adam
from layer import CSGNN


def Create_model(args):

    model = CSGNN(aggregator=args.aggregator, feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1, dropout=args.dropout)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer