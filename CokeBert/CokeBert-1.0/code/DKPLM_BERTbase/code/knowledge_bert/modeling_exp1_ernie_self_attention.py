def self_attention(self, Q, K, V, mode=None, ent_linear=None, ent_n=None):
    Q_l2 = Q.unsqueeze(1)
    Q_l2 = Q_l2.unsqueeze(2)
    K = ent_linear(K)
    K_l2 = K
    attention = (Q_l2 * K_l2).sum(3).div(math.sqrt(self.K_V_dim))
    attention = attention.masked_fill(attention == 0, float('-10000'))
    attention = self.softmax_2(self.LeakyReLU(attention))
    attention = attention.masked_fill(attention == float(1 / attention.
        shape[-1]), float(0))
    attention = attention.unsqueeze(2)
    if mode == 'entity':
        ans_list = list()
        for i_th, batch_i in enumerate(attention):
            for j_th, ent_ws in enumerate(batch_i):
                ent_j_T_or_F = ent_n[j_th] != 0
                mask_len = len(ent_j_T_or_F[ent_j_T_or_F == False])
                ent_ws = ent_ws[ent_ws != -100]
                ent_ans = list()
                mean_score = float(1 / (len(ent_ws) - mask_len))
                for k_th, ent_w in enumerate(ent_ws):
                    if float(ent_w) > mean_score:
                        ent_ans.append(1)
                    elif ent_j_T_or_F[k_th] == False:
                        ent_ans.append(-1)
                    else:
                        ent_ans.append(0)
                ans_list.append(ent_ans)
        return ans_list
