import os.path

import dgl
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_BATCH_SIZE = 64

# Step 1: Define Your Heterogeneous Graph Dataset
class GraphDataset(Dataset):
    def __init__(self, graph_list, labels):
        self.graph_list = graph_list
        self.labels = labels

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx], self.labels[idx]


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.5, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads   # 注意力键维度，需要整除
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))

        self.zero_relation_pri = nn.Parameter(torch.zeros(self.n_heads), requires_grad=False)
        self.zero_relation_att = nn.Parameter(torch.zeros(n_heads, self.d_k, self.d_k), requires_grad=False)
        self.zero_relation_msg = nn.Parameter(torch.zeros(n_heads, self.d_k, self.d_k), requires_grad=False)

        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        if edges.data['id'].size(0) == 0:
            relation_att = self.zero_relation_att
            relation_msg = self.zero_relation_msg
            relation_pri = self.zero_relation_pri
            # import pdb;pdb.set_trace()
            # return {'a': att, 'v': val}
        else:
            etype = edges.data['id'][0]
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
        key = torch.bmm(edges.src['k'].to(torch.float32).transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val = torch.bmm(edges.src['v'].to(torch.float32).transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]].to(torch.float64)
            v_linear = self.v_linears[node_dict[srctype]].to(torch.float64)
            q_linear = self.q_linears[node_dict[dsttype]].to(torch.float64)
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key].to(torch.float64)).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key].to(torch.float64)).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key].to(torch.float64)).view(-1, self.n_heads, self.d_k)

            # G.apply_edges(func=self.edge_attention, etype=etype)
            G.apply_edges(func=self.edge_attention,
                          etype=(srctype, etype, dsttype))
        # G.multi_update_all({etype: (self.message_func, self.reduce_func) \
        #                     for etype in edge_dict}, cross_reducer='mean')
        G.multi_update_all({(srctype, etype, dsttype): (self.message_func, self.reduce_func)
                            for srctype, etype, dsttype in G.canonical_etypes}, cross_reducer='mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            if 't' not in G.nodes[ntype].data:
                # G.nodes[ntype].data[out_key] = torch.Tensor(G.nodes[ntype].data['h'].size(0), self.out_dim).to(device)
                # G.nodes[ntype].data[out_key] = torch.Tensor(0, self.out_dim)
                continue
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'].to(torch.float32))
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1 - alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out.to(torch.float32)))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out.to(torch.float32))

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGTEncoder(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, n_layers, n_heads, num_node_types, num_edge_types, use_norm=True):
        super(HGTEncoder, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for _ in range(num_node_types):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, num_node_types, num_edge_types, n_heads,use_norm=use_norm))
        self.out = nn.Linear(n_hid, n_out, dtype=torch.float64)

    def forward(self, G):
        # node_reprs = []
        for ntype in G.ntypes:
            # n_id = G.get_ntype_id(ntype)
            n_id = G.node_dict[ntype]
            G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id].to(torch.float64)(G.nodes[ntype].data['inp']))
            # G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
            # node_reprs.append(G.nodes[ntype].data['h'])
        for i in range(self.n_layers):
            self.gcs[i](G, 'h', 'h')

        return G
        # if out_key == '__ALL__':
        #     return flatten_hete_ndata(G, 'h')
        # return G.nodes[out_key].data['h']
        # Generate graph representation by aggregating node representations (e.g., mean pooling)
        # You can also consider weighted aggregation based on the importance of each node type
        # graph_repr = torch.mean(torch.stack(node_reprs), dim=0)
        # output = self.out(graph_repr)
        # final_output = F.softmax(output, dim=1)
        # return final_output


class ClassficationDecoder(torch.nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_dim,
                 decoder_dp,
                 cls_num):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.decoder_dp = decoder_dp
        self.cls_num = cls_num
        self.linear1 = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.decoder_dim)
        self.bn1 = nn.InstanceNorm1d(self.decoder_dim)
        self.dp = nn.Dropout(self.decoder_dp)
        self.linear2 = nn.Linear(self.decoder_dim, self.cls_num)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = torch.einsum('bd->bd', x)
        if torch.prod(torch.tensor(x.dim())) > 1:
            x = self.bn(x)
        else:
            x = x.unsqueeze(0)
            x = self.bn1(x)
        x = self.dp(x)
        # x = torch.einsum('bdl->bld', x)
        x = self.linear2(x)
        return x


class GlobalAttentionPooling_hete(torch.nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def flatten_hete_ndata(self, out_G, node_key='h'):
        if type(out_G.ndata[node_key]) is not torch.Tensor:
            return torch.cat([out_G.ndata[node_key][k] for k in out_G.ndata[node_key]])
        else:
            return out_G.ndata[node_key]

    def forward(self, G):
        feat = self.flatten_hete_ndata(G)
        gate = self.gate_nn(feat)
        gate = F.softmax(gate, dim=0)
        out_r = feat * gate
        readout = torch.sum(out_r, dim=0, keepdim=True)
        return readout


class GlobalAttentionPooling_on_feat(torch.nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, feat):
        gate = self.gate_nn(feat)
        gate = F.softmax(gate, dim=1)
        out_r = feat * gate
        readout = torch.sum(out_r, dim=1, keepdim=True)
        # return torch.sum(readout)
        return readout


class HGTClassfication(torch.nn.Module):
    def __init__(self, cls_num, num_node_types, num_edge_types, embedding_dim=256,
                 encoder_hidden_size=2048, nlayer=8, nhead=4,  use_cuda=True, dropout=0.2):
        super().__init__()
        self.cls_num = cls_num
        # self.src_vocab = src_vocab
        # self.src_vocab_size = len(src_vocab)

        self.encoder_embedding_dim = embedding_dim
        self.encoder_dim = embedding_dim
        self.decoder_dim = embedding_dim * 2
        # self.decoder_dim = embedding_dim

        self.encoder_hidden_size = encoder_hidden_size

        self.encoder_nlayers = nlayer

        self.use_cuda = use_cuda
        # self.device = torch.device("cuda" if use_cuda else "cpu")

        self.encoder_dp = dropout
        self.decoder_dp = dropout

        self.encoder_nhead = nhead

        # self.node_dict = node_edge_dict['node']
        # self.edge_dict = node_edge_dict['edge']
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.encoder = HGTEncoder(self.encoder_embedding_dim, self.encoder_hidden_size, self.encoder_dim,
                                  self.encoder_nlayers, self.encoder_nhead,  self.num_node_types, self.num_edge_types,
                                  # self.src_vocab_size, self.device,
                                  use_norm=True)
        self.decoder = ClassficationDecoder(
            self.encoder_dim, self.decoder_dim, self.decoder_dp, self.cls_num)
        self.global_attn = GlobalAttentionPooling_on_feat(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dim, self.encoder_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.encoder_dim * 2, 1)
        ))

        self.loss_func = F.cross_entropy
        # self.loss_func = CategoricalCrossEntropyLoss()
        # https://huggingface.co/transformers/model_doc/bert.html?highlight=bertconfig#transformers.BertConfig
        self.initializer_range = 0.02
        self.init_weights()

    def init_weights(self):
        # self.encoder.to(self.device)
        # self.decoder.to(self.device)
        # https://huggingface.co/transformers/_modules/transformers/modeling_utils.html#PreTrainedModel
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def batch_different_length_tensor(self, tensors):
        """
        Args:
            tensors: a list of tensors with different length
        Returns:
            a tensor with size (len(tensors), max(lengths), ...)
        """
        max_len = max(map(lambda t: t.size(0), tensors))
        batch_tensor = torch.zeros(len(tensors), max_len, tensors[0].size(-1))
        # batch_tensor = torch.zeros(len(tensors), max_len)
        for i, t in enumerate(tensors):
            batch_tensor[i, :t.size(0), :] = t
            # batch_tensor[i, :t.size(0)] = t
        return batch_tensor.cuda() if self.use_cuda else batch_tensor

    # def forward(self, src_graphs, tgt_sents, src_inputs = None, tgt_inputs = None, out_key = ['identifier', '__ALL__'][1]):
    def forward(self, src_batch_graph, labels):
        encoder_graphs = []
        for graph in src_batch_graph:
            graph = graph.to(device)
            encoder_graphs_list = []
            encoder_graph = self.encoder(graph)
            for ntype in encoder_graph.ntypes:
                encoder_graphs_list.append(encoder_graph.nodes[ntype].data['h'])
            encoder_graph = torch.cat(encoder_graphs_list)
            encoder_graphs.append(encoder_graph)

        encoder_graphs = self.batch_different_length_tensor(
            encoder_graphs)

        encoder_graphs = self.global_attn(encoder_graphs).squeeze()
        logits = self.decoder(encoder_graphs)

        soft_logits = F.softmax(logits, dim=1)
        loss = self.loss_func(logits, labels)

        return loss, soft_logits

def HGTTrain(model, train_graphs, train_labels, optimizer, num_epochs, val_graphs, val_labels , save_path):
    model.to(device)
    # best_val_loss = float('inf')  # Initialize with a large value
    best_f1 = 0  # Initialize with a min value

    if os.path.exists(save_path):
        print("Exist model file， loading...")
        checkpoint = torch.load(save_path)
        # best_val_loss = checkpoint['best_val_loss']
        best_f1 = checkpoint['best_f1']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        model.train()
        loss_train = 0
        block_train = int(len(train_graphs) / _BATCH_SIZE) + 1
        # for i in range(block_train):
        for idx, i in enumerate(tqdm.tqdm(range(block_train), ncols=100)):
            begin = i * _BATCH_SIZE
            end = min((i + 1) * _BATCH_SIZE, len(train_graphs))
            if begin >= end:
                break
            X_batch = train_graphs[begin:end]
            y_batch = train_labels[begin:end]
            tensor_labels = [torch.from_numpy(arr).float() for arr in y_batch]
            y_batch = torch.stack(tensor_labels).squeeze().to(torch.int64).to(device)
            if torch.tensor(y_batch.dim()) < 1:
                y_batch = y_batch.reshape(1)
            loss, logits = model(X_batch, y_batch)
            # loss = criterion(y_pred, y_batch)
            loss_train += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item() * len(y_batch)

        loss_train /= len(train_graphs)

        model.eval()  # Switch to evaluation mode
        loss_val = 0
        correct = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        preds = []
        labels = []
        scores = []

        block_val = int(len(val_graphs) / _BATCH_SIZE) + 1
        for i in range(block_val):
            begin = i * _BATCH_SIZE
            end = min((i + 1) * _BATCH_SIZE, len(val_graphs))
            if begin >= end:
                break
            X_batch = val_graphs[begin:end]
            y_batch = val_labels[begin:end]
            tensor_labels = [torch.from_numpy(arr).float() for arr in y_batch]
            y_batch = torch.stack(tensor_labels).squeeze().to(torch.int64).to(device)
            if torch.tensor(y_batch.dim()) < 1:
                y_batch = y_batch.reshape(1)
            with torch.no_grad():
                loss, logits = model(X_batch, y_batch)
                # loss_val += loss.item() * len(y_batch)
                pred = logits.argmax(dim=1)
                score = logits[:, 1]
                # statistic.
                correct += int((pred == y_batch).sum())  # Check against ground-truth labels.
                preds.extend(pred.int().tolist())
                labels.extend(y_batch.int().tolist())
                scores.extend(score.tolist())
                true_positives += int(((pred == 1) & (y_batch == 1)).sum())
                false_positives += int(((pred == 1) & (y_batch == 0)).sum())
                false_negatives += int(((pred == 0) & (y_batch == 1)).sum())

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        auc = roc_auc_score(labels, scores)
        mcc = matthews_corrcoef(labels, preds)
        # loss_val /= len(val_graphs)

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, mcc: {mcc:.4f}")

        # Save the model if validation loss has decreased
        # if loss_val < best_val_loss:
        if f1 > best_f1:
            best_f1 = f1
            # best_val_loss = loss_val
            checkpoint = {
                # 'best_val_loss': best_val_loss,
                'best_f1': best_f1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)

    print("Training finished.")

def HGTTest(model, test_graphs, test_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    preds = []
    labels = []
    scores = []

    block_test = int(len(test_graphs) / _BATCH_SIZE) + 1
    for i in range(block_test):

        begin = i * _BATCH_SIZE
        end = min((i + 1) * _BATCH_SIZE, len(test_graphs))
        if begin >= end:
            break
        X_batch = test_graphs[begin:end]
        y_batch = test_labels[begin:end]
        tensor_labels = [torch.from_numpy(arr).float() for arr in y_batch]
        y_batch = torch.stack(tensor_labels).squeeze().to(torch.int64).to(device)
        if torch.tensor(y_batch.dim()) < 1:
            y_batch = y_batch.reshape(1)
        with torch.no_grad():
            loss, logits = model(X_batch, y_batch)
            pred = logits.argmax(dim=1)
            score = logits[:, 1]
            # statistic.
            correct += int((pred == y_batch).sum())  # Check against ground-truth labels.
            preds.extend(pred.int().tolist())
            labels.extend(y_batch.int().tolist())
            scores.extend(score.tolist())
            true_positives += int(((pred == 1) & (y_batch == 1)).sum())
            false_positives += int(((pred == 1) & (y_batch == 0)).sum())
            false_negatives += int(((pred == 0) & (y_batch == 1)).sum())

    acc = correct / len(test_graphs)  # Derive ratio of correct predictions.
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    auc = roc_auc_score(labels, scores)
    mcc = matthews_corrcoef(labels, preds)

    print('accuracy:' + str(acc))
    print('precision:' + str(precision))
    print('recall:' + str(recall))
    print('f1-score:' + str(f1))
    print('auc:' + str(auc))
    print('mcc:' + str(mcc))

    return acc, preds, labels