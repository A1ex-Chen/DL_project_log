def self_attention(self, Q, K, V, mode=None):
    Q = PCA_svd(Q, 100).cuda().half()
    Q_l2 = Q.unsqueeze(1)
    Q_l2 = Q_l2.unsqueeze(2)
    attention = (Q * K).sum(3).div(math.sqrt(self.K_V_dim))
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_2(self.LeakyReLU(attention))
    attention = attention.masked_fill(attention == float(1 / attention.
        shape[-1]), float(0))
    attention = attention.unsqueeze(2)
    """
        ##-------------------------
        if mode == "entity":
            #batch always 1 here
            ###
            ans_list = list()
            ###
            for i_th, batch_i in enumerate(attention):
                #print("in 1 times")
                for j_th, ent_ws in enumerate(batch_i):
                    print("=========")
                    print("{}:".format(j_th))
                    print("-------------")
                    ent_ws = ent_ws[ent_ws!=0]
                    ###
                    ent_ans=list()
                    ###
                    mean_score = float(1/len(ent_ws))
                    #print("mean:",mean_score)
                    for k_th, ent_w in enumerate(ent_ws):
                        print("{}:  {}".format(k_th,ent_w))
                        ###
                        #print(ent_w, mean_score)
                        if float(ent_w) > mean_score:
                            ent_ans.append(1)
                        else:
                            ent_ans.append(0)
                        ###
                    ans_list.append(ent_ans)

                print("=========")
            print("===============================")
            print("===============================")
            ##-------------------------

            print(ans_list)
            #exit()
            return ans_list
            """
    if mode == 'entity':
        ans_list = list()
        for i_th, batch_i in enumerate(attention):
            for j_th, ent_ws in enumerate(batch_i):
                ent_ws = ent_ws[ent_ws != -100]
                ent_ans = list()
                mean_score = float(1.2 / len(ent_ws))
                for k_th, ent_w in enumerate(ent_ws):
                    if float(ent_w) > mean_score:
                        ent_ans.append(1)
                    else:
                        ent_ans.append(0)
                ans_list.append(ent_ans)
        return ans_list