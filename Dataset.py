__author__ = 'Xiaochi'

import scipy.sparse as sp
import numpy as np
import json

class Data:
    #TODO: to complete dataset process
    def __init__(self, path=None, TrainTest = 'Train', debug = True):
        '''

        :param path:
        :param TrainTest:
        '''
        self.ent2Id = {}
        self.Id2ent = {}
        self.rel2Id = {}
        self.rel2vec = {}
        self.Id2rel = {}
        self.data = []
        self.Triples = []
        self.TrainTest = TrainTest
        self.TestEnt = []

        self.valid_batch = None

        self._batch = 0

        if TrainTest == 'Train':
            fent = open(path + 'entities.txt', 'rb')
            n = -1
            for line in fent:
                n += 1
                line = line.decode('utf-8').strip()
                self.ent2Id[line] = n
                self.Id2ent[n] = line
            fent.close()
            print n + 1
            print 'entities load finised...'

            f = open(path + 'SentEnt.txt', 'rb')
            for line in f:
                line = line.decode('utf-8').strip()
                Id = self.ent2Id[line]
                self.TestEnt.append(Id)
            f.close()

            frel = open(path + 'relations.txt', 'rb')
            n = -1
            for line in frel:
                n += 1
                line = line.decode('utf-8').strip()
                self.rel2Id[line] = n
                self.Id2rel[n] = line
            frel.close()
            print n + 1
            for rel in self.rel2Id:
                vec = sp.csr_matrix(([1.0], ([0], [self.rel2Id[rel]])), shape=(1, len(self.rel2Id)))
                self.rel2vec[rel] = vec
            print 'relations load finised...'
            n = 0
            fhop = open(path + 'hops-filtered.txt', 'rb')
            for line in fhop:
                if debug:
                    n += 1
                if n == 10000:
                    break
                line = line.decode('utf-8').strip()
                terms = line.split('\t')
                h = self.ent2Id[terms[0]]
                segs = terms[1].split(',')
                r = self.rel2vec[segs[0]]
                t = self.ent2Id[segs[1]]
                self.Triples.append((h, r, t))
            fhop.close()
            print 'hops load finished...'

            n = -1
            f = open(path + 'Train.txt', 'rb')
            for line in f:
                n += 1
                line = line.decode('utf-8').strip()
                js = json.loads(line)

                path_arr = js['path']
                path_data = []
                for p in path_arr:
                    path_vec = []
                    terms = p.split(' ')
                    for rel in terms:
                        relId = self.rel2Id[rel]
                        path_vec.append(relId)
                    path_data.append(path_vec)

                sent_feat, hId, tId = self.SentProcess(js)
                self.data.append((sent_feat, hId, tId, path_data))
            f.close()
            print 'triple process finished...'
        else:
            '''
            '''
            #TODO: load test data


    def  nextTrain(self):
        '''

        :return:
        '''
        if self._batch >= 1:
            raise StopIteration()
        sh = sp.csr_matrix((0, len(self.ent2Id)))
        st = sp.csr_matrix((0, len(self.ent2Id)))
        nst = sp.csr_matrix((0, len(self.ent2Id)))
        sent = []
        ppath = sp.csr_matrix((0, len(self.rel2Id)))
        npath = sp.csr_matrix((0, len(self.rel2Id)))
        ph = sp.csr_matrix((0, len(self.ent2Id)))
        pr = sp.csr_matrix((0, len(self.rel2Id)))
        pt = sp.csr_matrix((0, len(self.ent2Id)))
        nh = sp.csr_matrix((0, len(self.ent2Id)))
        nr = sp.csr_matrix((0, len(self.rel2Id)))
        nt = sp.csr_matrix((0, len(self.ent2Id)))

        tt = self.Triples
        idxs = np.random.random_integers(0, self.dataset_size - 1, self.b_size * self.mini_batch_size)
        for idx in idxs:
            # sent
            samp = self.data[idx]
            senth = samp[1]
            sentt = samp[2]
            nsentt = self.TestEnt[np.random.randint(0, len(self.TestEnt))]
            senth_vec = sp.csr_matrix(([1.0], ([0], [senth])), shape=(1, len(self.ent2Id)))
            sentt_vec = sp.csr_matrix(([1.0], ([0], [sentt])), shape=(1, len(self.ent2Id)))
            nsentt_vec = sp.csr_matrix(([1.0], ([0], [nsentt])), shape=(1, len(self.ent2Id)))
            sfeat = np.array([samp[0]])
            while self.model.sentht_eval(sfeat, senth_vec, sentt_vec, nsentt_vec) < 0:
                newidx = np.random.randint(0, self.dataset_size)
                samp = self.data[newidx]
                senth = samp[1]
                sentt = samp[2]
                nsentt = np.random.randint(0, len(self.ent2Id))
                senth_vec = sp.csr_matrix(([1.0], ([0], [senth])), shape=(1, len(self.ent2Id)))
                sentt_vec = sp.csr_matrix(([1.0], ([0], [sentt])), shape=(1, len(self.ent2Id)))
                nsentt_vec = sp.csr_matrix(([1.0], ([0], [nsentt])), shape=(1, len(self.ent2Id)))
                sfeat = np.array([samp[0]])

            # path
            p = samp[3]
            pid = np.random.randint(0, len(p))
            pp = p[pid]
            pp_vec = self.GetPathFeat(pp)
            npp = np.random.random_integers(0, len(self.rel2Id) - 1, np.random.randint(1, 4))
            np_vec = self.GetPathFeat(npp)

            # trip
            tripId = np.random.randint(0, len(tt))
            h, pr_vec, t = tt[tripId]
            nhId = np.random.randint(0, len(self.ent2Id))
            ntId = np.random.randint(0, len(self.ent2Id))
            nrId = np.random.randint(0, len(self.rel2Id))
            nTrip = (nhId, nrId, ntId)
            ph_vec = sp.csr_matrix(([1.0], ([0], [h])), shape=(1, len(self.ent2Id)))
            pt_vec = sp.csr_matrix(([1.0], ([0], [t])), shape=(1, len(self.ent2Id)))
            nh_vec, nr_vec, nt_vec = self.GetTripleFeat(nTrip)

            while self.model.trip_eval(ph_vec, pr_vec, pt_vec, nh_vec, nr_vec, nt_vec) < 0:
                tripId = np.random.randint(0, len(tt))
                h, pr_vec, t = tt[tripId]
                nhId = np.random.randint(0, len(self.ent2Id))
                ntId = np.random.randint(0, len(self.ent2Id))
                nrId = np.random.randint(0, len(self.rel2Id))
                nTrip = (nhId, nrId, ntId)
                ph_vec = sp.csr_matrix(([1.0], ([0], [h])), shape=(1, len(self.ent2Id)))
                pt_vec = sp.csr_matrix(([1.0], ([0], [t])), shape=(1, len(self.ent2Id)))
                nh_vec, nr_vec, nt_vec = self.GetTripleFeat(nTrip)

            sent.append(samp[0])
            sh = sp.vstack([sh, senth_vec])
            st = sp.vstack([st, sentt_vec])
            nst = sp.vstack([nst, nsentt_vec])
            ppath = sp.vstack([ppath, pp_vec])
            npath = sp.vstack([npath, np_vec])
            ph = sp.vstack([ph, ph_vec])
            pr = sp.vstack([pr, pr_vec])
            pt = sp.vstack([pt, pt_vec])
            nh = sp.vstack([nh, nh_vec])
            nr = sp.vstack([nr, nr_vec])
            nt = sp.vstack([nt, nt_vec])

        sent = np.array(sent)

        rval = (sent, sh, st, nst, ppath, npath, ph, pr, pt, nh, nr, nt)

        self._batch += 1

        return rval

    def netTest(self):
        '''

        :return:
        '''
        if self._batch >= 1:
            raise StopIteration()
        self._batch += 1
        rval = self.valid_batch

        return rval

    def GetPathFeat(self, path_arr):
        path_vec = sp.csr_matrix((1, len(self.rel2Id)))
        for relId in path_arr:
            r = sp.csr_matrix(([1.0], ([0], [relId])), shape=(1, len(self.rel2Id)))
            path_vec += r
        return path_vec

    def GetTripleFeat(self, triple):
        h, r, t = triple
        h_vec = sp.csr_matrix(([1.0], ([0], [h])), shape=(1, len(self.ent2Id)))
        r_vec = sp.csr_matrix(([1.0], ([0], [r])), shape=(1, len(self.rel2Id)))
        t_vec = sp.csr_matrix(([1.0], ([0], [t])), shape=(1, len(self.ent2Id)))
        return h_vec, r_vec, t_vec


