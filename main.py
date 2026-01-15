import os
import util
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd

from loader import CountyDataset
from parse import args
from model import  GraphEncoder


def evaluate(model, loader, device):
    model.eval()
    ys, hs, preds, pop, county, tms = [], [], [], [], [], []

    with torch.no_grad():
        for xs, sdi, edge, y, h, p, tm, cnty in loader:

            node_list = xs.to(args.device)
            sdi_list = sdi.to(args.device)
            edge_list = edge.to(args.device).long()

            x, sdi, edge_index, batch = merge_graph_batch(
                node_list, sdi_list, edge_list, device=args.device
            )

            y = y.to(args.device).float()
            pred = model(x, sdi, edge_index, batch)

            h = h.to(device).float()
            ys.append(y.cpu()* p / 10000)
            hs.append(h.cpu()* p / 10000)
            pop.append(p.cpu())
            county.append(cnty[0])
            tms.append(tm.numpy()[0])
            preds.append(pred.cpu()* p / 10000)

    ys = torch.cat(ys).numpy() 
    preds = torch.cat(preds).numpy()
    hs = torch.cat(hs).numpy()
    pop = torch.cat(pop).numpy()

    return util.smape(ys, hs), util.mae(ys, hs), util.rmse(ys, hs)

def merge_graph_batch(node_list, sdi_list, edge_list, device):
    all_nodes = []
    all_sdi = []
    all_edges = []
    batch_vector = []

    node_offset = 0 

    for graph_idx, (nodes, sdi, edges) in enumerate(zip(node_list, sdi_list, edge_list)):

        num_nodes = nodes.size(0)

        all_nodes.append(nodes)  
        all_sdi.append(sdi) 

        edges_shifted = edges + node_offset
        all_edges.append(edges_shifted)

        batch_vector.append(
            torch.full((num_nodes,), graph_idx, dtype=torch.long, device=device)
        )
        node_offset += num_nodes

    x = torch.cat(all_nodes, dim=0)            
    sdi = torch.cat(all_sdi, dim=0)      
    edge_index = torch.cat(all_edges, dim=1)     
    batch = torch.cat(batch_vector, dim=0)        

    return x, sdi, edge_index, batch


def main():

    df, sdi = util.read_county_data()
    train_loader = DataLoader(CountyDataset(df, sdi, args, phase='train'), batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(CountyDataset(df, sdi, args, phase='valid'), batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(CountyDataset(df, sdi, args, phase='test'), batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ========= Model =========
    model = GraphEncoder(
        input_dim=len(args.feature_cols),
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        graph_input_dim=len(args.feature_cols)
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    print("Start training …")

    # ========= Training loop =========
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for xs, sdi, edge, y, h, p, tm, cnty in tqdm(train_loader, desc=f"Epoch {epoch}"):
            node_list = xs.to(args.device)
            sdi_list = sdi.to(args.device)
            edge_list = edge.to(args.device).long()

            x, sdi, edge_index, batch = merge_graph_batch(
                node_list, sdi_list, edge_list, device=args.device
            )

            y = y.to(args.device).float()

            optimizer.zero_grad()
            pred = model(x, sdi, edge_index, batch)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_smape, train_mean_smape, _ = evaluate(model, train_loader, args.device)
        valid_smape, valid_mean_smape, _ = evaluate(model, valid_loader, args.device)
        test_smape, test_mae, test_rmse  = evaluate(model, test_loader,  args.device)

        print(
            f"Epoch {epoch:02d} | "
            f"TrainLoss={total_loss/len(train_loader):.4f} | "
            f"Train SMAPE={train_smape:.4f} | "
            f"Valid SMAPE={valid_smape:.4f} | "
            f"Test SMAPE={test_smape:.4f} | "
            f"Test MAE={test_mae:.4f} | "
            f"Test RMSE={test_rmse:.4f}"
        )

    # ========= Save model =========
    save_path = "model_overdose.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")


if __name__ == '__main__':
    main()
