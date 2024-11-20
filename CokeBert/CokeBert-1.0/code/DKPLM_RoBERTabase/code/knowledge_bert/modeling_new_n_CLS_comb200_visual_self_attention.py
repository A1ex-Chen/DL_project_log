def self_attention(self, Q, K_1, V_1, K_2, V_2, mode=None):
    """
        #Q
        Q = self.Q_linear(Q)
        Q = self.Tanh(Q)
        #[,100]
        Q = Q.unsqueeze(1)
        Q = Q.unsqueeze(2)
        Q = Q.repeat(1,K.shape[1],K.shape[2],1)

        #K
        K = self.V_linear(V)

        #V
        V = self.V_linear(V) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)

        #attention
        #print(Q.shape)
        #[2,2,100,100] ???
        #print(K.shape)
        attention = torch.norm(Q-K, p=1, dim=3, keepdim=True)
        mask = K.sum(dim=3)
        mask[mask!=0] = 1
        mask = mask.unsqueeze(3)
        #attention = mask*attention
        attention = mask.div(attention) #the more l1 loss, the more difference
        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_2(self.LeakyReLU(attention))
        #attention = self.softmax_2(self.LeakyReLU(attention))
        #print(attention)
        #print(attention.shape)
        attention = attention.squeeze(3).unsqueeze(2)

        sentence_entity_reps = attention.matmul(V).squeeze(2)
        """
    """
        Q_2 = self.K_V_linear_2(V_1)
        Q_2 = Q_2.unsqueeze(3)
        K_2 = self.K_V_linear_2(V_2)
        K_2 = torch.transpose(K_2,3,4)
        attention = (Q_2.matmul(K_2)).div(math.sqrt(self.K_V_dim))
        attention = attention.squeeze(3)

        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_3(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)).unsqueeze(-2) # don't need to

        #V_2 = self.V_linear_2(V_2) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)
        sentence_entity_reps = attention.matmul(V_2).squeeze(3)
        """
    """
        #V_1[:,:,0,:] = 0
        V_org_1 = V_1 - K_1
        V_org_1[:,:,0,:] = 0
        V_org_2 = V_2 - K_2
        V_org_1 = self.V_linear_2(V_org_1) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)
        V_org_2 = self.V_linear_2(V_org_2) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)
        V_org_2 = torch.transpose(V_org_2, 3, 4)
        attention = V_org_1.unsqueeze(2).matmul(V_org_2).sum(4).div(math.sqrt(self.K_V_dim))
        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_3(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to
        attention = attention.unsqueeze(3)
        V_2 = self.V_linear_2(V_2) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)
        sentence_entity_reps = attention.matmul(V_2).squeeze(3)
        """
    Q_2 = self.Q_linear_2(Q)
    Q_2 = self.Tanh(Q_2)
    Q_2 = Q_2.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    K = self.K_V_linear_2(K_2)
    attention = (Q_2 * K).sum(4).div(math.sqrt(self.K_V_dim))
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_3(self.LeakyReLU(attention))
    attention = attention.masked_fill(attention == float(1 / attention.
        shape[-1]), float(0))
    attention = attention.unsqueeze(3)
    sentence_entity_reps = attention.matmul(V_2).squeeze(3)
    if mode == 'candidate_pos':
        V_mask = V_1.sum(3)
        V_mask = V_mask.masked_fill(V_mask != 0, float(1)).unsqueeze(-1)
        sentence_entity_reps = V_mask * sentence_entity_reps
        V_1 = V_mask * V_1
        V_1 = torch.cat([V_1, sentence_entity_reps], -1)
    elif mode == 'candidate_neg':
        V_mask = V_1.sum(3)
        V_mask = (V_mask.masked_fill(V_mask != 0, float(1)).unsqueeze(-1) - 1
            ) * -1
        sentence_entity_reps = V_mask * sentence_entity_reps
        V_1 = V_mask * V_1
        V_1 = torch.cat([V_1, sentence_entity_reps], -1)
    else:
        V_1 = torch.cat([V_1, sentence_entity_reps], -1)
    Q_1 = self.Q_linear_1(Q)
    Q_1 = self.Tanh(Q_1)
    """
        Q_1 = F.normalize(Q_1, p=2, dim=1)
        Q_1 = Q_1.unsqueeze(1).unsqueeze(2)
        """
    Q_1 = Q_1.unsqueeze(1).unsqueeze(2)
    K = self.K_V_linear_1(K_1)
    """
        l2 = torch.norm(K, p=2, dim=3, keepdim=True).detach()
        l2[l2==0]=float(1e-6)
        K = K.div(l2)
        """
    """
        attention = (Q_1*K).sum(3)
        """
    attention = (Q_1 * K).sum(3).div(math.sqrt(self.K_V_dim))
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_2(self.LeakyReLU(attention))
    attention = attention.masked_fill(attention == float(1 / attention.
        shape[-1]), float(0))
    attention = attention.unsqueeze(2)
    sentence_entity_reps = attention.matmul(V_1).squeeze(2)
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
    return sentence_entity_reps
