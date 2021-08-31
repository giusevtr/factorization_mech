from dataloading.dataset import Dataset
from dataloading.domain import Domain
from hdmm.workload import Marginal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


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

def laplace_mech(dataset, epsilon, workload):
    marginal_queries = MyMarginals(dataset.domain, workload)
    answers = marginal_queries.get_answers(dataset)
    sensitivity = len(workload)/dataset.df.shape[0]
    #dataset[dataset[marginal_queries] == 1].shape[0] + np.random.laplace(loc=0, scale=sensitivity/epsilon)
    noise = np.random.laplace(loc=0, scale=sensitivity/epsilon, size=answers.shape)
    noisy_answers = answers + noise
    return answers, noisy_answers

def gaussian_mech(dataset, epsilon, workload, delta):
    marginal_queries = MyMarginals(dataset.domain, workload)
    answers = marginal_queries.get_answers(dataset)
    sensitivity = len(workload)/dataset.df.shape[0]
    #dataset[dataset[marginal_queries] == 1].shape[0] + np.random.laplace(loc=0, scale=sensitivity/epsilon)
    noise = np.random.laplace(loc=0, scale=sensitivity/epsilon, size=answers.shape)
    noisy_answers = [answers + noise for i in range(delta)]
    return answers, np.mean(noisy_answers)

def workloads():
    dataset = Dataset.load('data/adult.csv', 'data/adult-domain.json')
    # dataset = Dataset.load('data/test.csv', 'data/test-domain.json')
    data = pd.read_csv(r"/Users/nikhiljain/PycharmProjects/factorization_mech/data/adult-domain.json")
    data_attributes = []
    for elem in data:
        for i in range(len(elem)):
            if elem[i] == '"':
                if i >= 2:
                    data_attributes.append(elem[2:i])
    # print(data_attributes)
    list_of_workloads = []
    workload_lengths = [10]
    for n in workload_lengths:
        workload = []
        i = 0
        while i < n:
            elem_in_workload = []
            k = random.randint(1, 3)
            while len(elem_in_workload) < k:
                p = random.randint(0, 14)
                if data_attributes[p] not in elem_in_workload:
                    elem_in_workload.append(data_attributes[p])
            workload.append(elem_in_workload)
            i += 1
        list_of_workloads.append(workload)
    return list_of_workloads

def laplace_error_calc(dataset, workload):
    epsilon_values = [0.1, 0.2, 0.5, 1, 1.5, 2]
    error_values = []
    for elem in epsilon_values:
        print(elem)
        real_answers, laplace_answers = laplace_mech(dataset, elem, workload)
        error = np.linalg.norm(real_answers - laplace_answers)
        max_error = np.abs(real_answers - laplace_answers).max()
        print(elem, max_error)
        error_values.append(max_error)
    plt.plot(epsilon_values, error_values, 'o-')

def gaussian_error_calc(dataset, workload):
    epsilon_values = [0.1, 0.2, 0.5, 1, 1.5, 2]
    error_values = []
    for elem in epsilon_values:
        print(elem)
        real_answers, gaussian_answers = gaussian_mech(dataset, elem, workload, delta = 500)
        error = np.linalg.norm(real_answers - gaussian_answers)
        max_error = np.abs(real_answers - gaussian_answers).max()
        print(elem, max_error)
        error_values.append(max_error)
    plt.plot(epsilon_values, error_values, 'o-')


if __name__ == "__main__":
    dataset = Dataset.load('data/adult.csv', 'data/adult-domain.json')
    # # dataset = Dataset.load('data/test.csv', 'data/test-domain.json')
    data = pd.read_csv(r"/Users/nikhiljain/PycharmProjects/factorization_mech/data/adult-domain.json")
    # data_attributes = []
    # for elem in data:
    #     for i in range(len(elem)):
    #         if elem[i] == '"':
    #             if i >= 2:
    #                 data_attributes.append(elem[2:i])
    # print(data_attributes)
    # workload_lengths = [10, 20, 30, 40]
    # for n in workload_lengths:
    #     workload = []
    #     i = 0
    #     while i < n:
    #         elem_in_workload = []
    #         k = random.randint(1, 3)
    #         while len(elem_in_workload) < k:
    #             p = random.randint(0, 14)
    #             if data_attributes[p] not in elem_in_workload:
    #                 elem_in_workload.append(data_attributes[p])
    #         workload.append(elem_in_workload)
    #         i += 1
    #     print(workload)
    #     epsilon_values = [0.1, 0.2, 0.5, 1, 1.5, 2]
    #     error_values = []
    #     for elem in epsilon_values:
    #         print(elem)
    #         real_answers, laplace_answers = laplace_mech(dataset,elem,workload)
    #         error = np.linalg.norm(real_answers - laplace_answers)
    #         max_error = np.abs(real_answers - laplace_answers).max()
    #         print(elem, max_error)
    #         error_values.append(max_error)
    #     plt.plot(epsilon_values, error_values, 'o-', label = n)
    workloads_list = workloads()
    print(workloads_list)
    for elem in workloads_list:
        laplace_error_calc(dataset, elem)
        gaussian_error_calc(dataset, elem)
    plt.title('changes in error related to epsilon')
    plt.xlabel('epsilon')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    #print(laplace_answers)

    #print(dataset.df.head())

    #workload = [('A', 'B'), ('A', 'C'), ('C', 'D')]
    """
    A, B
    A=0, B=0
    A=0, B=1
    A=1, B=0
    A=1, B=1
    q(x) = [1, 0, 0, 0]
    """
    #marginal_queries = MyMarginals(dataset.domain, workload)

    #answers = marginal_queries.get_answers(dataset)

    #print(answers)
