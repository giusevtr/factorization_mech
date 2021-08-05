from dataloading.dataset import Dataset
from dataloading.domain import Domain
from hdmm.workload import Marginal
import numpy as np


class MyMarginal:
    def __init__(self, domain: Domain, proj: tuple):
        self.proj = proj
        self.proj_dom = domain.project(proj)
        self.W = Marginal.fromtuple(domain.shape, domain.axes(self.proj))

    def get_W_ith_row(self, query_id):
        q = np.zeros(self.proj_dom.size())
        q[query_id] = 1
        r = self.W._transpose().dot(q)
        return r

    def get_answers_from_weights(self, weights):
        return self.W.dot(weights)

    def get_answers_from_db(self, db: Dataset):
        N = db.df.shape[0]
        # print(self.W.shape)
        # return self.W.dot(db.project(self.proj).datavector() / N)
        return self.W.dot(db.datavector() / N)

    def sensitivity(self):
        return 1


class MyMarginals:
    def __init__(self, domain: Domain, workloads: list):
        self.workloads = workloads
        self.marginals = []
        self.query_id_to_marginal = {}
        query_id = 0
        for marginal_id, proj in enumerate(workloads):
            proj_dom = domain.project(proj)
            m = MyMarginal(domain, proj)
            self.marginals.append(m)

            for j in range(proj_dom.size()):
                self.query_id_to_marginal[query_id] = (marginal_id, j)
                query_id = query_id + 1

    def get_error(self, real: Dataset, fake: Dataset):
        return np.max(np.abs(self.get_answers(real) - self.get_answers(fake)))

    def get_error_weights(self, real_vec, fake_vec):
        return np.max(np.abs(self.get_answers_weights(real_vec) - self.get_answers_weights(fake_vec)))

    # def get_answers(self, db: Dataset, concat=True):
    #     n = len(db.df)
    #     wei = db.datavector(flatten=False) / n
    #     return self.get_answers_weights(wei, concat)

    def get_answers(self, data, weights=None, concat=True, debug=False):
        ans_vec = []
        N_sync = data.df.shape[0]
        # for proj, W in self.workloads:
        for proj in self.workloads:
            # weights let's you do a weighted sum
            x = data.project(proj).datavector(weights=weights)
            if weights is None:
                x = x / N_sync
            ans_vec.append(x)
        if concat:
            ans_vec = np.concatenate(ans_vec)
        return ans_vec

    def get_answers_weights(self, weights, concat=True):
        diff = [mar.W.dot(weights) for mar in self.marginals]
        if not concat:
            return diff
        return np.concatenate(diff)

def gaussian_mech(marginas_queries, dataset, epsilon):
    ## Laplace mechanism
    """
    for all D, D'
    for any S
    pr(M(D) \in S )\leq e^{epsilon} pr(M(D') \in S)
    """

    # get sensitivity:
    # query sensitivity: q(D) = sum_{x \in D } q(x)
    # |q(D) - q(D')| \leq 1

    # l1-sensitivity of f is  max_{D,D'} || f(D) - f(D') ||_1 .`
    #
    # v  = [q_1(D), q_2(D), ..., q_m(D)]
    # v' = [q_1(D'), q_2(D'), ..., q_m(D')]
    # || v - v' ||_1  <= workload_size

    real_answers = marginas_queries.get_answers(dataset)
    priv_answers = None
    return priv_answers


if __name__ == "__main__":
    # dataset = Dataset.load('data/adult.csv', 'data/adult-domain.json')
    dataset = Dataset.load('data/test.csv', 'data/test-domain.json')

    print(dataset.df.head())

    workload = [('A', 'B'), ('A', 'C'), ('C', 'D')]
    """
    A, B
    A=0, B=0
    A=0, B=1
    A=1, B=0
    A=1, B=1
    q(x) = [1, 0, 0, 0]
    """
    marginal_queries = MyMarginals(dataset.domain, workload)

    answers = marginal_queries.get_answers(dataset)

    print(answers)