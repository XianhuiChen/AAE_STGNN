import pandas as pd
import numpy as np
import torch, json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CountyDataset(Dataset):
    """
    Dataset for overdose prediction.
    Each sample is one (zcta, time) feature vector,
    and the label is N months later overdose death count.
    """
    def __init__(self, merged_df, sdoh, args, level='County', phase='train'):
        self.area_tm_dict = {}
        self.phase = phase
        self.area_edge_index_dict = json.load(open('./data/county_neighbors.json'))
        sdoh.fillna(0, inplace=True)
        for _, row in tqdm(merged_df.iterrows()):
            if level == 'County':
                area = row['County']
            else:
                area = row['ZCTA']
            tm = int(row['tm'])
            sdi = sdoh[sdoh[level] == area]
            
            sdi_col = sdi[args.sdoh_cols].values[0] *0.01 if len(sdi) > 0 else np.zeros(len(args.sdoh_cols))

            if area not in self.area_tm_dict:
                self.area_tm_dict[area] = {}
            if tm not in self.area_tm_dict[area]:
                ccs = [f'{c}.pos_rolling_3' for c in args.feature_cols]
                fcs = [f'{c}.per_rolling_3' for c in args.feature_cols]
                ccs[-1] = 'population'
                fcs[-1] = 'label.per'
                
                self.area_tm_dict[area][tm] = {
                    'feature': np.array(row[fcs].values, dtype=np.float32),
                    'count': np.array(row[ccs].values, dtype=np.float32),
                    'label_hist': row['opioid.overdose'],
                    'sdi': sdi_col,
                    'label': row['label_rolling_3'],
                    'population': row['population'],
                }

        self.st_list = []
        for area in tqdm(self.area_tm_dict):
            tm_list = sorted(self.area_tm_dict[area].keys())
            for tm in tm_list[args.observation_window - 1:- args.prediction_window]:
                if phase == 'train' and tm > args.train_tm:
                    continue
                if phase == 'test' and tm < args.valid_tm:
                    continue
                if phase == 'valid' and (tm > args.valid_tm or tm < args.train_tm):
                    continue
                # if phase=='train' or self.area_tm_dict[area][tm]['population'] > 300000:
                if self.area_tm_dict[area][tm]['population'] > 0:
                    self.st_list.append([area, tm])

        self.args = args

    def __len__(self):
        return len(self.st_list)


    def random_augmentation(self, idx):
        x, s, y, h, p, c, tm, area = self.read_idx(idx)


        # area, tm = self.st_list[idx]
        area_set = set([area])
        xs = [x]
        sdi = [s]
        ys = [y]
        hs = [h]
        ps = [p]
        cs = [c]
        while sum(ps) < 300000:
            ars = list(self.area_tm_dict.keys())
            np.random.shuffle(ars)
            for area in ars:
                if area not in area_set:
                    if tm in self.area_tm_dict[area]:
                        area_set.add(area)
                        break
            x, s, y, h, p, c, _, _ = self.read_idx(area=area, tm=tm)
            if p < 1:
                print('p', p, 'y', y, 'h', h)
                continue
            # if c.min() < 1:
            #     print('p', p, 'y', y, 'h', h)
            #     continue
            xs.append(x)
            sdi.append(s)
            ys.append(y)
            hs.append(h)
            ps.append(p)
            cs.append(c)

        nh = sum([p * h for p, h in zip(ps, hs)]) / sum(ps)
        ny = sum([p * y for p, y in zip(ps, ys)]) / sum(ps)
        sdi = np.mean(np.stack(sdi), axis=0)
        nx = 0
        norm = 0.0001
        for x,c in zip(xs, cs):
            nx = nx + x * c
            norm = norm + c
        nx = nx / norm
        
        # print(nx.shape, nx.dtype, xs[0].shape, xs[0].dtype, norm.shape, norm.dtype, cs[0].shape, cs[0].dtype)
        # print(ny.shape, ny.dtype, ys[0].shape, ys[0].dtype)
        # print(nh.shape)
        return nx, sdi, ny, nh, sum(ps), tm, area

    def __getitem__(self, idx):
        # print(self.use_gnn)
        # if self.use_gnn:
        node_list = []
        sdi_list = []
        area, tm = self.st_list[idx]
        x, s, yt, ht, pt = self.read_idx(idx)[:5]
        # node_list.append(x.reshape(-1))
        node_list.append(x)
        sdi_list.append(s)
        neighbors = self.area_edge_index_dict[area]
        for neighbor in neighbors:
            try:
                x, s, y, h = self.read_idx(area=neighbor, tm=tm)[:4]
            except:
                continue
            # node_list.append(x.reshape(-1))
            node_list.append(x)
            sdi_list.append(s)

        edge_list = []
        num_nodes = len(node_list)
        # center node index = 0
        center_idx = 0
        for i in range(1, num_nodes):
            edge_list.append([center_idx, i])
            edge_list.append([i, center_idx])  # 无向图加反向边



        # for n in node_list:
        #     print(n.dtype, n.shape)
        # node_list = np.array(node_list)
        node_list = [torch.tensor(n, dtype=torch.float32) for n in node_list]
        node_list = torch.stack(node_list)
        sdi_list = [torch.tensor(s, dtype=torch.float32) for s in sdi_list]
        sdi_list = torch.stack(sdi_list)
        # print('x', node_list.shape)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return node_list, sdi_list, edge_index, yt, ht, pt, tm, area
        # elif self.phase != 'train':
        #     x, s, yt, ht, pt, _, tm, cnty = self.read_idx(idx)
        #     edge = 0
        #     return x, s, edge, yt, ht, pt, tm, cnty
        # else:
        #     # return self.read_idx(idx)[:3]
        #     x, s, yt, ht, pt, tm, cnty = self.random_augmentation(idx)
        #     edge = 0
        #     return x, s, edge, yt, ht, pt, tm, cnty


    def read_idx(self, idx=-1, area=None, tm=-1):
        if area is None:
            area, tm = self.st_list[idx]
        else:
            pass

        xs = []  # 用来堆叠每个月的 feature
        cs = []
        hs = []  # 用来计算历史 overdose label

        # 取 observation_window 内的特征
        for i in range(self.args.observation_window):
            cur_tm = tm - i
            if cur_tm in self.area_tm_dict[area]:
                xs.append(torch.from_numpy(self.area_tm_dict[area][cur_tm]['feature']).float())
                cs.append(torch.from_numpy(self.area_tm_dict[area][cur_tm]['count']).float())
                h = torch.tensor(self.area_tm_dict[area][cur_tm]['label_hist']).float()
                hs.append(h)
            else:
                # 如果某个月缺数据，就补 zeros (保证维度一致)
                xs.append(torch.zeros(len(self.args.feature_cols)))
                cs.append(torch.zeros(len(self.args.feature_cols)))

        # xs 顺序目前是 [tm, tm-1, tm-2 ...]，需要倒序 => 时间从过去到现在
        xs = torch.stack(xs[::-1])
        cs = torch.stack(cs[::-1])

        # 历史 overdose 的平均值
        h = torch.mean(torch.stack(hs))

        # 当前时间点的 label
        try:
            y = torch.tensor(self.area_tm_dict[area][tm]['label']).float()
        except:
            # print(self.st_list[idx], area, tm, sorted(self.area_tm_dict[area].keys()))
            pass

        y = torch.tensor(self.area_tm_dict[area][tm]['label']).float()
        # print('y', y, 'area',  area, 'tm', tm)
        p = self.area_tm_dict[area][tm]['population']
        sdi = self.area_tm_dict[area][tm]['sdi']
        y = y / p * 10000

        h = h / p * 10000
        assert p > 0

        return xs, sdi, y, h, p, cs, tm, area


