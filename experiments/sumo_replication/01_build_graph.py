#!/usr/bin/env python3
"""Deterministic connected METR-LA *abstract* graph, no OSM or sensor coordinates."""
import argparse, json, os, sys, subprocess, shutil, hashlib, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from config import ROOT, CODE
sys.path.insert(0, str(CODE))


def select_subgraph(adj, n, k):
    a = np.array(adj, dtype=float, copy=True)
    np.fill_diagonal(a, 0)
    if a.shape[0] < n: raise ValueError(f'N={n} exceeds METR-LA graph size {len(a)}')
    # Symmetric top-k graph; this is the declared sparsification, not claimed road topology.
    w = np.maximum(a, a.T)
    w[~np.isfinite(w)] = 0.
    sparse = np.zeros_like(w)
    for i in range(len(w)):
        cand = np.flatnonzero(w[i] > 0)
        for j in sorted(cand, key=lambda j: (-w[i,j],j))[:k]:
            sparse[i,j] = w[i,j]
    sparse = np.maximum(sparse,sparse.T)
    degree = (sparse > 0).sum(1)
    starts = sorted(range(len(a)), key=lambda i:(-degree[i],i))
    for start in starts:
        nodes, seen, front = [], {start}, [start]
        while front and len(nodes)<n:
            i = front.pop(0); nodes.append(i)
            cand = np.flatnonzero(sparse[i] > 0)
            for j in sorted(cand, key=lambda j:(-sparse[i,j],j)):
                if j not in seen: seen.add(int(j)); front.append(int(j))
        if len(nodes)==n:
            sub=sparse[np.ix_(nodes,nodes)]
            return np.array(nodes,int),sub
    raise RuntimeError('Cannot select requested connected component; lower --nodes or increase --topk')


def build_network(root, sub, freeflow_mph):
    nodes,edges=ET.Element('nodes'),ET.Element('edges')
    n=len(sub)
    # Purely abstract ring layout, not geographic coordinates.
    xy=np.column_stack((450*np.cos(2*np.pi*np.arange(n)/n),450*np.sin(2*np.pi*np.arange(n)/n)))
    for i,(x,y) in enumerate(xy):
        ET.SubElement(nodes,'node',id=f'in{i}',x=f'{x-9:.3f}',y=f'{y:.3f}',type='priority')
        ET.SubElement(nodes,'node',id=f'out{i}',x=f'{x+9:.3f}',y=f'{y:.3f}',type='priority')
        ET.SubElement(edges,'edge',id=f'sensor_{i}',**{'from':f'in{i}','to':f'out{i}'},
                      priority='2',numLanes='1',speed=f'{freeflow_mph/2.2369362920544:.4f}')
    for i in range(n):
        for j in range(n):
            if i!=j and sub[i,j]>0:
                ET.SubElement(edges,'edge',id=f'link_{i}_{j}',**{'from':f'out{i}','to':f'in{j}'},
                  priority='1',numLanes='1',speed=f'{freeflow_mph/2.2369362920544:.4f}')
    for filename,obj in [('nodes.nod.xml',nodes),('edges.edg.xml',edges)]:
        ET.ElementTree(obj).write(root/filename,encoding='utf-8',xml_declaration=True)
    exe=shutil.which('netconvert')
    if not exe: raise RuntimeError('netconvert is missing; activate SUMO_HOME/bin in PATH')
    cmd=[exe,'--node-files',str(root/'nodes.nod.xml'),'--edge-files',str(root/'edges.edg.xml'),
         '--output-file',str(root/'abstract.net.xml'),'--no-turnarounds','true',
         '--junctions.corner-detail','0','--no-warnings','true']
    cp=subprocess.run(cmd,capture_output=True,text=True)
    if cp.returncode or not (root/'abstract.net.xml').exists():
        raise RuntimeError(f'netconvert rc={cp.returncode}: {cp.stderr[-4000:]}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--nodes',type=int,default=24)
    p.add_argument('--topk',type=int,default=4);p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    from causalgap.data import load_dataset
    bundle=load_dataset('METR-LA',history_len=12,pred_len=12)
    idx,sub=select_subgraph(bundle.adj,args.nodes,args.topk)
    # Recover original mph. Train period only, not validation/test.
    tr=bundle.scaler.inverse(bundle.train_x[...,0])[:,:,idx]
    tr=tr[np.isfinite(tr)&(tr>0)]
    freeflow=float(np.clip(np.quantile(tr,.85),20,75))
    sensor_ids=[]
    # Dataset may not expose ID order; source h5 columns define canonical order when possible.
    p_h5=CODE.parent/'datasets'/'METR-LA'/'metr-la.h5'
    if p_h5.exists():
        h=pd.read_hdf(p_h5)
        if len(h.columns)==bundle.num_nodes: sensor_ids=[str(h.columns[int(i)]) for i in idx]
    if not sensor_ids: sensor_ids=[f'model_node_{int(i)}' for i in idx]
    build_network(args.root,sub,freeflow)
    np.savez_compressed(args.root/'subgraph.npz',indices=idx,adj=sub.astype('float32'),
             sensor_ids=np.asarray(sensor_ids),freeflow_mph=np.float32(freeflow))
    meta=dict(protocol='abstract_topology_only_no_osm_no_real_od',N=len(idx),topk=args.topk,
      selection='highest-degree connected BFS on symmetrized METR-LA top-k adjacency',
      model_node_indices=idx.tolist(),sensor_ids=sensor_ids,freeflow_mph=freeflow,
      network='abstract.net.xml',coordinates='abstract_ring_not_geographic')
    (args.root/'graph_meta.json').write_text(json.dumps(meta,indent=2))
    print('[PASS] abstract graph N=',len(idx),'network=',args.root/'abstract.net.xml')
if __name__=='__main__':main()
