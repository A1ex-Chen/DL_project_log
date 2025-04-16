def self_attention(self, Q, K, V, fun):
    Q = self.Q_linear(Q)
    Q = self.Tanh(Q)
    Q = Q.unsqueeze(1)
    Q = Q.unsqueeze(2)
    Q = Q.repeat(1, K.shape[1], K.shape[2], 1)
    K = self.V_linear(K)
    V = self.V_linear(V)
    attention = torch.norm(Q - K, p=1, dim=3, keepdim=True)
    mask = K.sum(dim=3)
    mask[mask != 0] = 1
    mask = mask.unsqueeze(3)
    attention = mask.div(attention)
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_2(self.LeakyReLU(attention))
    if fun == 'test':
        for i_th, batch_i in enumerate(attention):
            for j_th, ent_ws in enumerate(batch_i):
                print('=========')
                print('{}:'.format(j_th))
                print('-------------')
                ent_ws = ent_ws[ent_ws != 0]
                for k_th, ent_w in enumerate(ent_ws):
                    print('{}:  {}'.format(k_th, ent_w))
            print('=========')
        print('===============================')
        print('===============================')
    attention = attention.squeeze(3).unsqueeze(2)
    sentence_entity_reps = attention.matmul(V).squeeze(2)
    try:
        sentence_entity_reps = attention.matmul(V).squeeze(2)
    except:
        sentence_entity_reps = torch.zeros(V.shape[0], V.shape[1], 1, 100)
    """
        ####Matrix####
        #[--]
        #Q

        #[--]
        Q = self.Q_linear(Q)
        Q = self.Tanh(Q)
        #[--]
        Q_l2 = F.normalize(Q, p=2, dim=1)
        Q_l2 = Q_l2.unsqueeze(1)
        Q_l2 = Q_l2.unsqueeze(2)
        #Q_l2 = F.normalize(Q, p=2, dim=2) #
        #Q = Q.reshape(Q.shape[0]*Q.shape[1],Q.shape[2]) #

        #Q query V:
        #K = self.K_V_linear(K)
        #K = self.K_V_linear(V)
        #K = self.V_linear(K)
        K = self.V_linear(V)
        #print(self.V_linear.bias)
        #exit()
        #K = self.Tanh(K)

        l2 = torch.norm(K, p=2, dim=3, keepdim=True).detach()
        l2[l2==0]=float(1e-6)
        K_l2 = K.div(l2)
        #print(K_l2)
        #print(K_l2.shape)
        attention = (Q_l2*K_l2).sum(3)
        #print(attention)
        #exit()

        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_2(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0))


        ################################
        if fun=="test":
            #print(attention)
            #print(attention.shape)
            #[batches, entities, neighbor]
            for i_th, batch_i in enumerate(attention):
                #print("in")
                #print(batch_i)
                #print(batch_i.shape)
                #exit()
                for j_th, ent_ws in enumerate(batch_i):
                    #print("-------------")
                    print("=========")
                    print("{}:".format(j_th))
                    print("-------------")
                    ent_ws = ent_ws[ent_ws!=0]
                    for k_th, ent_w in enumerate(ent_ws):
                        print("{}:  {}".format(k_th,ent_w))
                print("=========")
            print("===============================")
            print("===============================")
            #exit()
        ################################

        attention = attention.unsqueeze(2)


        V = self.V_linear(V) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)
        #V = self.Tanh(V)
        #V = K
        try:
            sentence_entity_reps = attention.matmul(V).squeeze(2)
        except:
            sentence_entity_reps = torch.zeros(V.shape[0],V.shape[1],1,100)

        """
    return sentence_entity_reps
