import pandas as pd 
# from fim import fpgrowth,fim 
import numpy as np
import math
from itertools import chain, combinations
import itertools
from numpy.random import random
from bisect import bisect_left
from random import sample
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
import time
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
# from scipy.special import comb


class BOA(object):
    def __init__(self, binary_data,Y):
        self.df = binary_data  
        self.Y = Y
        self.attributeLevelNum = defaultdict(int) 
        self.attributeNames = []
        for i,name in enumerate(binary_data.columns):
          attribute = name.split('_')[0]
          self.attributeLevelNum[attribute] += 1
          self.attributeNames.append(attribute)
        self.attributeNames = list(set(self.attributeNames))
    
    def set_parameters(self, a1=1000,b1=1,a2=1000,b2=1,alpha_l=None,beta_l = None):
        # input al and bl are lists
        self.alpha_1 = a1
        self.beta_1 = b1
        self.alpha_2 = a2
        self.beta_2 = b2
        if alpha_l==None:
            self.alpha_l = [1 for i in range(self.maxlen+1)]
        else:
            self.alpha_l = alpha_l
        if beta_l ==None:
            self.beta_l= [(self.patternSpace[i]*100+1) for i in range(self.maxlen+1)]
        else:
            self.beta_l = beta_l

    def getPatternSpace(self):
        # print 'Computing sizes for pattern space ...'
        self.patternSpace = list(np.bincount(self.rules_len, minlength = self.maxlen+1))

        # start_time = time.time()
        # """ compute the rule space from the levels in each attribute """
        # for item in self.attributeNames:
        #     self.attributeLevelNum[item+'_neg'] = self.attributeLevelNum[item]
        # self.patternSpace = np.ones(self.maxlen+1)
        # tmp = [ item + '_neg' for item in self.attributeNames]
        # self.attributeNames.extend(tmp)
        # for k in range(1,self.maxlen+1,1):
        #     for subset in combinations(self.attributeNames,k):
        #         tmp = 1
        #         for i in subset:
        #             tmp  *=  self.attributeLevelNum[i]
        #         self.patternSpace[k] +=  tmp
        # print('\tTook %0.3fs to compute patternspace' % (time.time() - start_time))               

    def precompute(self):
        # compute L0
        TP,FP,TN,FN = sum(self.Y),0,len(self.Y) - sum(self.Y),0
        self.Lup = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)+ log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
        self.const_denominator = [np.log(np.true_divide(self.patternSpace[l]+self.beta_l[l]-1,self.patternSpace[l]+self.alpha_l[l]-1)) for l in range(self.maxlen+1)]
        Kn_count = [0 for l in range(self.maxlen+1)]
        self.P0 =  sum([log_betabin(Kn_count[i],self.patternSpace[i],self.alpha_l[i],self.beta_l[i]) for i in range(1,len(Kn_count),1)])            

# This function generates rules that satisfy supp and maxlen using fpgrowth, then it selects the top N rules that make data have the biggest decrease in entropy
# there are two ways to generate rules. fpgrowth can handle cases where the maxlen is small. If maxlen<=3, fpgrowth can generates rules much faster than randomforest. 
# If maxlen is big, fpgrowh tends to generate too many rules that overflow the memories. 
    def generate_rules(self,supp,maxlen,N, need_negcode = False,njobs = 5, method = 'rf ',criteria = 'IG',add_rules = []):
        self.maxlen = maxlen
        df = self.df
        if method =='fpgrowth':
            print('Using fpgrowth to generate rules ...')
            if need_negcode: # if df does not have negations 
                df = 1-self.df 
                df.columns = [name.strip() + 'neg' for name in self.df.columns]
                df = pd.concat([self.df,df],axis = 1)
            itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]  
            self.itemMatrix = itemMatrix
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
            start_time = time.time()
            rules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            rules = [np.sort(x[0]).tolist() for x in rules]
        else:
            print('NEW -- Using random forest to generate rules ...')
            rules = []
            start_time = time.time()
            for length in range(1,maxlen+1,1):
                n_estimators = 500*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n],df.columns))
            rules = [list(x) for x in set(tuple(np.sort(x)) for x in rules)]   
            df = 1-self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df,df],axis = 1)

        self.generate_time = time.time() - start_time
        print('\tTook %0.3fs to generate %d rules' % (self.generate_time, len(rules)))
        index = []
        for rule in add_rules:
            if np.sort(rule).tolist()  not in rules:
                rules.append(rule)
                index.append(len(rules)-1)
            else:
                index.append(rules.index(rule))

        self.screen_rules(rules,df,N,supp,criteria,njobs,index) # select the top N rules using secondary criteria, information gain
        self.getPatternSpace()

    def screen_rules(self,rules,df,N,supp,criteria = 'IG',njobs = 5,add_rules = []):
        # print 'screening rules'
        start_time = time.time()
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        mat = np.matrix(df) * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)

        Zpos = [Z[i] for i in np.where(self.Y>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        supp_select = np.where(TP>=supp*sum(self.Y)/100)[0]
        if len(supp_select)<=N:
            self.rules = [rules[i] for i in supp_select]
            self.RMatrix = np.array(Z[:,supp_select])
            self.rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in self.rules]
            self.supp = np.array(np.sum(Z,axis=0).tolist()[0])[supp_select]
        else:
            FP = np.array(np.sum(Z,axis = 0))[0] - TP
            TN = len(self.Y) - np.sum(self.Y) - FP
            FN = np.sum(self.Y) - TP
            p1 = TP.astype(float)/(TP+FP)
            p2 = FN.astype(float)/(FN+TN)
            pp = (TP+FP).astype(float)/(TP+FP+TN+FN)

            if criteria =='precision':
                select = np.argsort(p1[supp_select])[::-1][:N].tolist()
            # elif criteria = 'fisher':
            #     FP = np.array(np.sum(Z,axis = 0))[0] - TP
            #     TN = len(self.Y) - np.sum(self.Y) - FP
            #     FN = np.sum(self.Y) - TP
            else: # 
                cond_entropy = -pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1))-(1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2))
                cond_entropy[p1*(1-p1)==0] = -((1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2)))[p1*(1-p1)==0]
                cond_entropy[p2*(1-p2)==0] = -(pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1)))[p2*(1-p2)==0]
                cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
                pos = (TP+FN).astype(float)/(TP+FP+TN+FN)
                info = - pos * np.log(pos) - (1-pos)*np.log(1-pos)
                info[np.where((pos==1)| (pos==0))[0]] = 0
                IGR = (info - cond_entropy)/info 
                IGR[np.where(info==0)[0]] = 0
                select = np.argsort(IGR[supp_select])[::-1][:N].tolist()
            ind = list(supp_select[select])
            self.rules = [rules[i] for i in ind]
            self.RMatrix = np.array(Z[:,ind])
            self.rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in self.rules]
            self.supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind]
        self.screen_time = time.time() - start_time
        # print '\tTook %0.3fs to generate %d rules' % (self.screen_time, len(self.rules))




    def SA_patternbased(self, Niteration = 5000, Nchain = 3, print_message=True,q = 0.1, init = []):
        # print 'Searching for an optimal solution...'
        self.precompute()
        start_time = time.time()
        # print 'Asize0 = {}'.format(self.Asize0)
        self.Asize = [[min(self.patternSpace[l]/2,0.5*(self.patternSpace[l]+self.beta_l[l]-self.alpha_l[l])) for l in range(self.maxlen+1)]]
        self.C = [1]

        nRules = len(self.rules)
        maps = defaultdict(list)
        T0 = 1000
        # split = 0.6* Niteration
        elapsed_time = []

        for chain in range(Nchain):
            # initialize with a random pattern set
            if init !=[]:
                rules_curr = init[:]
            else:
                N = 1#sample(list(range(1,min(20,nRules),1)),1)[0]
                rules_curr = sample(list(range(nRules)),N)
            rules_curr_norm = self.normalize(rules_curr)
            pt_curr = -1000000000
            maps[chain].append([-1,[pt_curr/3,pt_curr/3,pt_curr/3],rules_curr,[self.rules[i] for i in rules_curr],[]])

            for iter in range(Niteration):
                # if iter>=split:
                #     p = np.array(range(1+len(maps[chain])))
                #     p = np.array(list(accumulate(p)))
                #     p = p/p[-1]
                #     index = find_lt(p,random())
                #     rules_curr = maps[chain][index][2][:]
                #     rules_curr_norm = maps[chain][index][2][:]
                rules_new, rules_norm = self.propose(rules_curr, rules_curr_norm,q)
                cfmatrix,prob =  self.compute_prob(rules_new)
                T = T0**(1 - iter/Niteration)
                pt_new = sum(prob)
                alpha = np.exp(float(pt_new -pt_curr)/T)
                
                if pt_new > sum(maps[chain][-1][1]) :
                    if print_message:
                        print('\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}\n old is {}, pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n '.format(chain, iter,(cfmatrix[0]+cfmatrix[2]+0.0)/len(self.Y),cfmatrix[0],cfmatrix[1],cfmatrix[2],cfmatrix[3],sum(maps[chain][-1][1])+0.1,sum(prob), prob[0], prob[1], prob[2]))
                        self.print_rules(rules_new)
                        print(rules_new)
                    self.Asize.append([np.floor(min(self.Asize[-1][l],np.true_divide(-pt_new + self.Lup + self.P0,self.const_denominator[l])))  for l in range(self.maxlen+1)])
                    # print 'Asize is {}, vt:{} = {}'.format(sum(self.Asize[-1][1:]),pt_new,prob)
                    self.const_denominator = [np.log(np.true_divide(self.patternSpace[l]+self.beta_l[l]-1,max(1,self.Asize[-1][l]+self.alpha_l[l]-1))) for l in range(self.maxlen+1)]
                    # print '\n** chain = {}, max at iter = {} ** \n obj = {}'.format(chain, iter,pt_new)
                    maps[chain].append([iter,prob,rules_new,[self.rules[i] for i in rules_new],cfmatrix])
                    new_supp = np.ceil(np.log(max([np.true_divide(self.patternSpace[l]-self.Asize[-1][l]+self.beta_l[l],max(1,self.Asize[-1][l]-1+self.alpha_l[l])) for l in range(1,self.maxlen+1,1)])))
                    self.C.append(new_supp)
                if random() <= alpha:
                    rules_curr_norm,rules_curr,pt_curr = rules_norm[:],rules_new[:],pt_new
                if np.mod(iter+1,10)==0:
                    elapsed_time.append(time.time() - start_time)
        pt_max = [sum(maps[chain][-1][1]) for chain in range(Nchain)]
        index = pt_max.index(max(pt_max))
        elapsed_time = time.time() - start_time
        # print '\tTook %0.3fs to generate an optimal rule set' % (t)
        return maps[index][-1][3],maps[index]

    # def SA_patternbased_experiment(self, Niteration = 5000, Nchain = 3, q = 0.1, init = [], print_message=False):
    #     # print 'Searching for an optimal solution...'
    #     self.precompute()
    #     start_time = time.time()
    #     # print 'Asize0 = {}'.format(self.Asize0)
    #     self.Asize = [[min(self.patternSpace[l]/2,0.5*(self.patternSpace[l]+self.beta_l[l]-self.alpha_l[l])) for l in range(self.maxlen+1)]]

    #     nRules = len(self.rules)
    #     maps = defaultdict(list)
    #     T0 = 1000
    #     split = 0.7*Niteration
    #     for chain in range(Nchain):
    #         # initialize with a random pattern set
    #         if init !=[]:
    #             rules_curr = init[:]
    #         else:
    #             N = sample(range(1,min(8,nRules),1),1)[0]
    #             rules_curr = sample(range(nRules),N)
    #         rules_curr_norm = self.normalize(rules_curr)
    #         pt_curr = -10000
    #         maps[chain].append([-1,[pt_curr/3,pt_curr/3,pt_curr/3],rules_curr,[self.rules[i] for i in rules_curr]])

    #         for iter in range(Niteration):
    #             if iter>=split:
    #                 p = np.array(range(1+len(maps[chain])))
    #                 p = np.array(list(accumulate(p)))
    #                 p = p/p[-1]
    #                 index = find_lt(p,random())
    #                 rules_curr = maps[chain][index][2][:]
    #                 rules_curr_norm = maps[chain][index][2][:]
    #             cfmatrix,prob =  self.compute_prob(rules_new)
    #             T = T0**(1 - iter/Niteration)
    #             pt_new = sum(prob)
    #             alpha = np.exp(float(pt_new -pt_curr)/T)
    #             maps[chain].append([iter,prob,rules_new,[self.rules[i] for i in rules_new]])

    #             if pt_new > sum(maps[chain][-1][1]) :
    #                 if print_message:
    #                     print '\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}\n old is {}, pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n '.format(chain, iter,(cfmatrix[0]+cfmatrix[2]+0.0)/len(self.Y),cfmatrix[0],cfmatrix[1],cfmatrix[2],cfmatrix[3],sum(maps[chain][-1][1])+0.1,sum(prob), prob[0], prob[1], prob[2])
    #                     # print '\n** chain = {}, max at iter = {} ** \n obj = {}, prior = {}, llh = {} '.format(chain, iter,prior+llh,prior,llh)
    #                     self.print_rules(rules_new)
    #                     print rules_new
    #                 self.Asize.append([np.floor(min(self.Asize[-1][l],np.true_divide(-pt_new + self.Lup + self.P0,self.const_denominator[l])))  for l in range(self.maxlen+1)])
    #                 print 'Asize is {}, vt:{} = {}'.format(sum(self.Asize[-1][1:]),pt_new,prob)
    #                 self.const_denominator = [np.log(np.true_divide(self.patternSpace[l]+self.beta_l[l]-1,max(1,self.Asize[-1][l]+self.alpha_l[l]-1))) for l in range(self.maxlen+1)]
    #                 print '\n** chain = {}, max at iter = {} ** \n obj = {}'.format(chain, iter,pt_new)
    #             if random() <= alpha:
    #                 rules_curr_norm,rules_curr,pt_curr = rules_norm[:],rules_new[:],pt_new
    #     pt_max = [sum(maps[chain][-1][1]) for chain in range(Nchain)]
    #     index = pt_max.index(max(pt_max))
    #     # print '\tTook %0.3fs to generate an optimal rule set' % (time.time() - start_time)
    #     self.maps = maps
    #     return maps[index][-1][3]

    def propose(self, rules_curr,rules_norm,q):
        nRules = len(self.rules)
        Yhat = (np.sum(self.RMatrix[:,rules_curr],axis = 1)>0).astype(int)
        incorr = np.where(self.Y!=Yhat)[0]
        N = len(rules_curr)
        if len(incorr)==0:
            clean = True
            move = ['clean']
            # it means the HBOA correctly classified all points but there could be redundant patterns, so cleaning is needed
        else:
            clean = False
            ex = sample(list(incorr),1)[0]
            t = random()
            if self.Y[ex]==1 or N==1:
                if t<1.0/2 or N==1:
                    move = ['add']       # action: add
                else:
                    move = ['cut','add'] # action: replace
            else:
                if t<1.0/2:
                    move = ['cut']       # action: cut
                else:
                    move = ['cut','add'] # action: replace
        if move[0]=='cut':
            """ cut """
            if random()<q:
                candidate = list(set(np.where(self.RMatrix[ex,:]==1)[0]).intersection(rules_curr))
                if len(candidate)==0:
                    candidate = rules_curr
                cut_rule = sample(candidate,1)[0]
            else:
                p = []
                all_sum = np.sum(self.RMatrix[:,rules_curr],axis = 1)
                for index,rule in enumerate(rules_curr):
                    Yhat= ((all_sum - np.array(self.RMatrix[:,rule]))>0).astype(int)
                    TP,FP,TN,FN  = getConfusion(Yhat,self.Y)
                    p.append(TP.astype(float)/(TP+FP+1))
                    # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p,0,0)
                p = np.array(list(accumulate(p)))
                if p[-1]==0:
                    index = sample(list(range(len(rules_curr))),1)[0]
                else:
                    p = p/p[-1]
                index = find_lt(p,random())
                cut_rule = rules_curr[index]
            rules_curr.remove(cut_rule)
            rules_norm = self.normalize(rules_curr)
            move.remove('cut')
            
        if len(move)>0 and move[0]=='add':
            """ add """
            if self.Y[ex]==1:
                select = list(set(np.where(self.supp>self.C[-1])[0]).intersection(np.where(self.RMatrix[ex,:]==1)[0]))
            else:
                select = list(set(np.where(self.supp>self.C[-1])[0]).intersection(np.where(self.RMatrix[ex,:]==0)[0]))
            if len(select)>0:
                if random()<q:
                    add_rule = sample(select,1)[0]
                else: 
                    Yhat_neg_index = np.where(np.sum(self.RMatrix[:,rules_curr],axis = 1)<1)[0]
                    self.Yhat_neg_index = Yhat_neg_index
                    mat = np.multiply(self.RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),self.Y[Yhat_neg_index])
                    # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                    TP = np.sum(mat,axis = 1)
                    FP = np.array(np.sum(self.RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                    TN = np.sum(self.Y[Yhat_neg_index]==0)-FP
                    FN = sum(self.Y[Yhat_neg_index]) - TP
                    p = (TP.astype(float)/(TP+FP+1))
                try:
                    add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
                    if add_rule not in rules_curr:
                        rules_curr.append(add_rule)
                except:
                    1

        if len(move)>0 and move[0]=='clean':
            remove = []
            for i,rule in enumerate(rules_norm):
                Yhat = (np.sum(self.RMatrix[:,[rule for j,rule in enumerate(rules_norm) if (j!=i and j not in remove)]],axis = 1)>0).astype(int)
                TP,FP,TN,FN = getConfusion(Yhat,self.Y)
                if TP+FP==0:
                    remove.append(i)
            for x in remove:
                rules_norm.remove(x)
            return rules_curr, rules_norm
        return rules_curr, rules_norm

    def compute_prob(self,rules):
        Yhat = (np.sum(self.RMatrix[:,rules],axis = 1)>0).astype(int)
        TP,FP,TN,FN = getConfusion(Yhat,self.Y)
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength = self.maxlen+1))
        prior_ChsRules= sum([log_betabin(Kn_count[i],self.patternSpace[i],self.alpha_l[i],self.beta_l[i]) for i in range(1,len(Kn_count),1)])            
        # rho1 = 0.99999999999999999
        # rho2 = 0.51
        # likelihood_1 = TP *np.log(rho1) + FP * np.log(1 - rho1) #log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)
        # likelihood_2 = TN *np.log(rho2) + FN * np.log(1 - rho2)  #log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
        likelihood_1 = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)
        likelihood_2 = log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
        # precision = float(TP)/(TP + FP)
        # recall = float(TP)/(TP + FN)
        return [TP,FP,TN,FN],[prior_ChsRules,likelihood_1,likelihood_2]
        # return [TP,FP,TN,FN],[-len(rules)*0.02,float(TP)/len(Yhat),float(TN)/len(Yhat)]

    # def normalize_add(self, rules_new, rule_index):
    #     rules = rules_new[:]
    #     for rule in rules_new:
    #         if set(self.rules[rule]).issubset(self.rules[rule_index]):
    #             return rules_new[:]
    #         if set(self.rules[rule_index]).issubset(self.rules[rule]):
    #             rules.remove(rule)
    #     rules.append(rule_index)
    #     return rules

    def normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1<len(rules):
                for p2 in range(p1+1,len(rules),1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules[:]
        except:
            return rules_new[:]


    def print_rules(self, rules_max):
        for rule_index in rules_max:
            print(self.rules[rule_index])

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0


def log_betabin(k,n,alpha,beta):
    import math
    try:
        Const =  math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print('alpha = {}, beta = {}'.format(alpha,beta))
    if isinstance(k,list) or isinstance(k,np.ndarray):
        if len(k)!=len(n):
            print('length of k is %d and length of n is %d'%(len(k),len(n)))
            raise ValueError
        lbeta = []
        for ki,ni in zip(k,n):
            # lbeta.append(math.lgamma(ni+1)- math.lgamma(ki+1) - math.lgamma(ni-ki+1) + math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
            lbeta.append(math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
        return np.array(lbeta)
    else:
        return math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const

def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN

def predict(rules,df):
    Z = [[] for rule in rules]
    dfn = 1-df #df has negative associations
    dfn.columns = [name.strip() + 'neg' for name in df.columns]
    df = pd.concat([df,dfn],axis = 1)
    for i,rule in enumerate(rules):
        Z[i] = (np.sum(df[list(rule)],axis=1)==len(rule)).astype(int)
    Yhat = (np.sum(Z,axis=0)>0).astype(int)
    return Yhat

def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = 'neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)   
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules

def binary_code(df,collist,Nlevel):
    for col in collist:
        for q in range(1,Nlevel,1):
            threshold = df[col].quantile(float(q)/Nlevel)
            df[col+'_geq_'+str(int(q))+'q'] = (df[col] >= threshold).astype(float)
    df.drop(collist,axis = 1, inplace = True)


def compute_ml(A,n1,n2,alphal,betal,alpha1,beta1,alpha2,beta2):
    top =  log_betabin(n1,n1,alpha1,beta1)+log_betabin(n2,n2,alpha2,beta2)- log_betabin(0,0,alpha1,beta1)-log_betabin(n2,n2+n1,alpha2,beta2)
    bottom = [np.log(np.true_divide(A[l]+betal[l]-1,A[l]+alphal[l]-1)) for l in range(len(A))]
    return [np.true_divide(top,x) for x in bottom]

def compute_ml_tight(A,n1,n2,alphal,betal,alpha1,beta1,alpha2,beta2,pt_new):
    ml = A[:]
    top =  log_betabin(n1,n1,alpha1,beta1)+log_betabin(n2,n2,alpha2,beta2)- log_betabin(0,0,alpha1,beta1)-log_betabin(n2,n2+n1,alpha2,beta2)
    temp = top[:]
    c = 0
    while c<20:
        c += 1
        bottom = [np.log(np.true_divide(A[l]+betal[l]-1,temp[l]+alphal[l]-1)) for l in range(len(A))]
        temp = [np.true_divide(top,x) for x in bottom]
        if sum(abs(np.array(temp) - np.array(ml))) < 10:
            return temp
        else:
            # print(temp)
            ml = temp[:]

def compute_supp(A,m,n1,n2,alphal,betal,alpha1,beta1,alpha2,beta2):
    top = np.log(max([np.true_divide(A[l]-m[l]+betal[l],m[l]-1+alphal[l]) for l in range(len(A))][1:]))
    bottom = np.log(np.true_divide((n1+alpha1-1)*(n2+alpha2+beta2),(n1+alpha1+beta1-1)*(beta2)))
    return np.true_divide(top,bottom)
    