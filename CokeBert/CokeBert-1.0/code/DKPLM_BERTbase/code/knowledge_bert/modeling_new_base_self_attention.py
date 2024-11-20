def self_attention(self, Q, K, V):
    """
        #Q
        Q = self.Q_linear(Q)
        Q = self.Tanh(Q)
        #[,100]
        Q = Q.unsqueeze(1)
        Q = Q.unsqueeze(2)
        Q = Q.repeat(1,K.shape[1],K.shape[2],1)

        #K
        K = self.K_V_linear(K)

        #V
        V = self.V_linear(V) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)

        #attention
        #print(Q.shape)
        #[2,2,100,100] ???
        attention = torch.norm(Q-K, p=1, dim=3, keepdim=True)
        mask = K.sum(dim=3)
        mask[mask!=0] = 1
        mask = mask.unsqueeze(3)
        #attention = mask*attention
        attention = attention + float(1e-6)
        attention = mask.div(attention) #the more l1 loss, the more difference
        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_2(self.LeakyReLU(attention))
        #attention = self.softmax_2(self.LeakyReLU(attention))
        #print(attention)
        #print(attention.shape)
        attention = attention.squeeze(3).unsqueeze(2)

        sentence_entity_reps = attention.matmul(V).squeeze(2)
        """
    Q = self.Q_linear(Q)
    Q = self.Tanh(Q)
    Q_l2 = Q.unsqueeze(1)
    Q_l2 = Q_l2.unsqueeze(2)
    K = self.V_linear(K)
    K_l2 = K
    attention = (Q_l2 * K_l2).sum(3).div(math.sqrt(self.K_V_dim))
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_2(self.LeakyReLU(attention))
    attention = attention.masked_fill(attention == float(1 / attention.
        shape[-1]), float(0))
    attention = attention.unsqueeze(2)
    V = self.V_linear(V)
    sentence_entity_reps = attention.matmul(V).squeeze(2)
    """
        mask = V.sum(dim=3)
        mask[mask!=0]=1
        mask[:,:,0]=0
        #print(mask)
        #print(mask.shape)
        try:
            mask[0][1][:]=0
        except:
            mask = torch.zeros(8,2,100).cuda().half()
            print(mask)
            print(mask.shape)
        attention = mask.masked_fill(mask==0, float('-10000'))
        attention = self.softmax_2(attention)
        #attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) #[1,1,1,1] fail --> don't need to
        attention = attention.unsqueeze(2)
        sentence_entity_reps = attention.matmul(V).squeeze(2)
        #print(sentence_entity_reps)
        #print(sentence_entity_reps.shape)
        #sentence_entity_reps = V[:,:,0,:]
        """
    return sentence_entity_reps
