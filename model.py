import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def attention(query, key, value, mask=None, dropout=None):
    dim = key.shape[2]
    d_k = dim ** 0.5
    scores = torch.matmul(query, key.permute([0, 2, 1])) / d_k

    if mask is not None:
        adder = (1.0 - torch.cast(mask, torch.float32)) * -10000.0
        scores += adder
    p_attn = F.softmax(scores)
    if dropout is not None:
        p_attn = F.dropout(p_attn, dropout)
    return torch.matmul(p_attn, value), p_attn

class multihead_attention(nn.Module):
    def __init__(self, numunits, num_heads, dropout, mask, batchsize):
        super(multihead_attention, self).__init__()
        self.numunits = numunits
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask = mask
        self.batchsize = batchsize
        self.linear_Q = nn.Linear(self.numunits, self.numunits)
        self.linear_K = nn.Linear(self.numunits, self.numunits)
        self.linear_V = nn.Linear(self.numunits, self.numunits)
        self.linear_Q3 = nn.Linear(self.numunits, self.numunits)
        self.linear_K3 = nn.Linear(self.numunits, self.numunits)
        self.linear_V3 = nn.Linear(self.numunits, self.numunits)
        self.linear_out = nn.Linear(self.numunits, self.numunits)
        self.linear_Q4 = nn.Linear(self.numunits, self.numunits)
        self.linear_V4 = nn.Linear(self.numunits, self.numunits)
        self.linear_out5 = nn.Linear(4*self.numunits, self.numunits)

    def forward(self, queries1, queries2, queries3, adj):
        keys1 = queries1
        keys2 = queries2
        keys3 = queries3
        queries4 = torch.zeros_like(queries3)
        for i in range(128):
            queries4[i] = queries3[adj[i].long()]
        Q1 = self.linear_Q(queries1)
        Q1 = F.tanh(Q1)
        Q2 = self.linear_Q(queries2)
        Q2 = F.tanh(Q2)
        Q3 = self.linear_Q3(queries3)
        Q3 = F.tanh(Q3)
        Q4 = self.linear_Q4(queries4)
        Q4 = F.tanh(Q4)
        K3 = self.linear_K3(keys3)
        K3 = F.tanh(K3)
        V1 = self.linear_V(keys1)
        V1 = F.tanh(V1)
        V2 = self.linear_V(keys2)
        V2 = F.tanh(V2)
        V3 = self.linear_V3(keys3)
        V3 = F.tanh(V3)
        V4 = self.linear_V4(queries4)
        V4 = F.tanh(V4)
        Q1 = torch.split(Q1,self.num_heads,dim=-1)
        Q1_ = torch.cat(Q1, dim=0)
        Q2 = torch.split(Q2, self.num_heads, dim=-1)
        Q2_ = torch.cat(Q2, dim=0)
        Q3 = torch.split(Q3, self.num_heads, dim=-1)
        Q3_ = torch.cat(Q3, dim=0)
        Q4 = torch.split(Q4, self.num_heads, dim=-1)
        Q4_ = torch.cat(Q4, dim=0)
        K3 = torch.split(K3, self.num_heads, dim=-1)
        K3_ = torch.cat(K3, axis=0)
        V1 = torch.split(V1, self.num_heads,dim=-1)
        V1_ = torch.cat(V1, axis=0)
        V2 = torch.split(V2, self.num_heads, dim=-1)
        V2_ = torch.cat(V2, axis=0)
        V3 = torch.split(V3, self.num_heads, dim=-1)
        V3_ = torch.cat(V3, axis=0)
        V4 = torch.split(V4, self.num_heads, dim=-1)
        V4_ = torch.cat(V4, axis=0)
        outputs1, atten1 = attention(Q1_, K3_, V1_, mask=self.mask, dropout=self.dropout)
        outputs2, atten2 = attention(Q2_, K3_, V2_, mask=self.mask, dropout=self.dropout)
        outputs3, atten3 = attention(Q3_, K3_, V3_, mask=self.mask,
                                     dropout=self.dropout)
        outputs4, atten4 = attention(Q4_, K3_, V4_, mask=self.mask,
                                     dropout=self.dropout)
        outputs1 = torch.split(outputs1, self.batchsize, dim=0)
        outputs1 = torch.cat(outputs1, dim=-1)
        outputs2 = torch.split(outputs2, self.batchsize, dim=0)
        outputs2 = torch.cat(outputs2, dim=-1)
        outputs3 = torch.split(outputs3, self.batchsize, dim=0)
        outputs3 = torch.cat(outputs3, dim=-1)
        outputs4 = torch.split(outputs4, self.batchsize, dim=0)
        outputs4 = torch.cat(outputs4, dim=-1)
        outputs5 = torch.cat([outputs1,outputs2,outputs3,outputs4], dim = 2)
        outputs5 = self.linear_out5(outputs5)
        outputs5 = F.tanh(outputs5)

        return outputs1, outputs2, outputs5, outputs3

class transformer_encoder(nn.Module):
    def __init__(self, prembed_dim, num_heads, num_hidden, dropout_rate, attention_dropout, mask, batchsize):
        super(transformer_encoder, self).__init__()
        self.prembed_dim = prembed_dim
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.mask = mask
        self.is_training = True
        self.reuse = None
        self.batchsize = batchsize
        self.multihead_attention = multihead_attention(self.prembed_dim, self.num_heads, self.attention_dropout, self.mask, self.batchsize)
        self.feed1 = nn.Conv1d(in_channels=self.prembed_dim, out_channels=self.num_hidden, kernel_size=1)
        self.feed2 = nn.Conv1d(in_channels=self.num_hidden, out_channels=self.prembed_dim, kernel_size=1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, emb_b, emb_a, emb_c, name_scope, adj):
        embs_b, embs_a, embs_c, embs_glo= self.multihead_attention(emb_b, emb_a, emb_c, adj)
        embs_b1 = emb_b + embs_b
        embs_a1 = emb_a + embs_a
        embs_c1 = emb_c + embs_c
        embs_b2 = self.dropout(embs_b1)
        embs_a2 = self.dropout(embs_a1)
        embs_c2 = self.dropout(embs_c1)
        embs_b3 = F.layer_norm(embs_b2,[self.prembed_dim,])
        embs_a3 = F.layer_norm(embs_a2, [self.prembed_dim, ])
        embs_c3 = F.layer_norm(embs_c2, [self.prembed_dim, ])

        trans_embs_b1 = self.feed1(embs_b3.permute([0,2,1]))
        trans_embs_a1 = self.feed1(embs_a3.permute([0, 2, 1]))
        trans_embs_c1 = self.feed1(embs_c3.permute([0, 2, 1]))

        trans_embs_b2 = trans_embs_b1.permute([0,2,1])
        trans_embs_a2 = trans_embs_a1.permute([0, 2, 1])
        trans_embs_c2 = trans_embs_c1.permute([0, 2, 1])

        trans_embs1 = F.relu(trans_embs_b2)
        trans_embs2 = F.relu(trans_embs_a2)
        trans_embs3 = F.relu(trans_embs_c2)

        trans_embs1 = self.feed2(trans_embs1.permute([0,2,1]))
        trans_embs2 = self.feed2(trans_embs2.permute([0, 2, 1]))
        trans_embs3 = self.feed2(trans_embs3.permute([0, 2, 1]))
        trans_embs1 = trans_embs1.permute([0, 2, 1])
        trans_embs2 = trans_embs2.permute([0, 2, 1])
        trans_embs3 = trans_embs3.permute([0, 2, 1])
        embs1 = embs_b3 + trans_embs1
        embs2 = embs_a3 + trans_embs2
        embs3 = embs_c3 + trans_embs3
        embs1 = self.dropout(embs1)
        embs2 = self.dropout(embs2)
        embs3 = self.dropout(embs3)
        embs1 = F.layer_norm(embs1,[self.prembed_dim,])
        embs2 = F.layer_norm(embs2, [self.prembed_dim, ])
        embs3 = F.layer_norm(embs3, [self.prembed_dim, ])
        return embs1,  embs2, embs3

class TriTF(nn.Module):

    def __init__(self, max_len: int, n_channel: int, max_depth: int, num_head: int,
                 num_hidden: int, drop_rate: float, attention_dropout: float,
                prembed: bool, prembed_dim: int, masking: bool, pooling: bool, pool_size: int, batchsize: int):
        super(TriTF, self).__init__()
        self.max_len = max_len
        self.n_channel = n_channel
        self.max_depth = max_depth
        self.num_head = num_head
        self.drop_rate = drop_rate
        self.attention_dropout = attention_dropout
        self.prembed = prembed
        self.prembed_dim = prembed_dim
        self.num_hidden = num_hidden
        self.masking = masking
        self.pooling = pooling
        self.pool_size = pool_size
        self.batchsize = batchsize
        self.transformer_encoder = transformer_encoder(self.prembed_dim, self.num_head, self.num_hidden, self.drop_rate, self.attention_dropout, None, self.batchsize)
        self.PE_linear = nn.Linear(self.n_channel, self.prembed_dim)
        self.out_linear = nn.Linear(self.prembed_dim * self.max_len, 2)
        self.PE_linear3 = nn.Linear(self.n_channel, self.prembed_dim)
        self.bn = nn.BatchNorm1d(49)

    def forward(self, x1, x2, x_slic, adj):
        num_data = x1.shape[0]
        emb_c1 = torch.cat([x1,x2,x_slic],dim=2)
        emb_c2 = self.bn(emb_c1)
        emb_b_1 = emb_c2[:,:,:self.n_channel]
        emb_a_1 = emb_c2[:,:,self.n_channel:2*self.n_channel]
        emb_c2 = emb_b_1 - emb_a_1
        emb_b_1 = self.PE_linear(emb_b_1)
        emb_a_1 = self.PE_linear(emb_a_1)
        emb_c_1 = self.PE_linear3(emb_c2)
        emb_b_2 = F.softmax(emb_b_1,-1)
        emb_a_2 = F.softmax(emb_a_1, -1)
        emb_c_2 = F.softmax(emb_c_1, -1)

        for i in range(self.max_depth):
            name_scope = "Transformer_Encoder_" + str(i)
            if i == 0:
                enc_embs1, enc_embs2, enc_embs3 = self.transformer_encoder(emb_b_2, emb_a_2, emb_c_2, name_scope,adj)
            else:
                enc_embs1, enc_embs2, enc_embs3 = self.transformer_encoder(enc_embs1, enc_embs2, enc_embs3, name_scope,adj)

        enc_embs333 = enc_embs3.reshape([num_data, self.max_len * self.prembed_dim])
        logit = self.out_linear(enc_embs333)
        return logit