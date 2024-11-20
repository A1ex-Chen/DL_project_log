def self_attention(self, Q, K, V):
    """
        if self.K_weight.type() == "torch.cuda.HalfTensor":
            sentence_entity_reps = torch.FloatTensor(len(e_neighs),self.K_V_dim).half().cuda()
        else:
            sentence_entity_reps = torch.FloatTensor(len(e_neighs),self.K_V_dim).cuda()

        embed = torch.nn.Embedding.from_pretrained(self.embed) #
        #[--]
        #Q
        #[|]
        #Q = self.Q_weight.mm(Q.T) #
        Q = self.Q_linear(Q)
        Q = Q.T
        #[--]
        Q_l2 = F.normalize(Q.T, p=2, dim=1)
        #[|]
        Q_l2 = Q_l2.T
        for i,e_neigh in enumerate(e_neighs):
            #[--]
            if self.K_weight.type() == "torch.cuda.HalfTensor":
                K = embed(torch.LongTensor(e_neigh)).half().cuda() #
            else:
                K = embed(torch.LongTensor(e_neigh)).cuda() #

            #K = self.K_weight.mm(K.T).T #
            K = self.K_V_linear(K)
            #print(K)
            #print(K.shape)
            K_l2 = F.normalize(K, p=2, dim=1)
            #print(K_l2)
            #print(K_l2.shape)

            ##[|]
            #Q_l2 = self.Q_weight.mm(Q.T) #
            ##[--]
            #Q_l2 = F.normalize(Q_l2.T, p=2, dim=1)
            ##[|]
            #Q_l2 = Q_l2.T

            attention = self.softmax_0( torch.sum(torch.matmul(K_l2,Q_l2), dim=1, out=None)) #
            #attention = self.LeakyReLU( torch.sum(torch.matmul(K_l2,Q_l2), dim=1, out=None)) #
            sentence_entity_rep = attention.matmul(K)
            sentence_entity_reps[i][:] = sentence_entity_rep
        """
    Q = self.Q_linear(Q)
    Q_l2 = F.normalize(Q, p=2, dim=1)
    Q_l2 = Q_l2.unsqueeze(1)
    Q_l2 = Q_l2.unsqueeze(2)
    """
        K = self.K_V_linear(K)
        #K_l2 = F.normalize(K, p=2, dim=2, eps=1e-6)
        #l2 = torch.norm(K, p=2, dim=2, keepdim=True).detach()
        l2 = torch.norm(K, p=2, dim=3, keepdim=True).detach()
        #l2 = l2+float(1e-6)
        l2[l2==0]=float(1e-6)
        K_l2 = K.div(l2)
        """
    K = self.V_linear(V)
    l2 = torch.norm(K, p=2, dim=3, keepdim=True).detach()
    l2[l2 == 0] = float(1e-06)
    K_l2 = K.div(l2)
    attention = (Q_l2 * K_l2).sum(3)
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_2(self.LeakyReLU(attention))
    attention = attention.masked_fill(attention == float(1 / attention.
        shape[-1]), float(0))
    attention = attention.unsqueeze(2)
    V = self.V_linear(V)
    sentence_entity_reps = attention.matmul(V).squeeze(2)
    return sentence_entity_reps
