
from model.LambdaRank.lambdarank import LambdaRank
from model.RankNet.ranknet import RankNet

def get_ranking_model(model, para):
    if model  == 'RankNet':
        ranker = RankNet(para)

    elif model =='LambdaRank':
        ranker = LambdaRank(para)

    return ranker

