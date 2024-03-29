import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2
import time
import random 

def foreground2background(dis, obj_num):
    if obj_num == 1:
        return dis
    bg_dis = []
    for i in range(obj_num):
        obj_back = []
        for j in range(obj_num):
            if i == j:
                continue
            obj_back.append(dis[j].unsqueeze(0))
        obj_back = torch.cat(obj_back, dim=1)
        obj_back, _ = torch.min(obj_back, dim=1, keepdim=True)
        bg_dis.append(obj_back)
    bg_dis = torch.cat(bg_dis, dim=0)
    return bg_dis

WRONG_LABEL_PADDING_DISTANCE = 5e4
#############################################################GLOBAL_DIST_MAP
def _pairwise_distances(x, x2, y, y2):
    #print("x",x.size())
    #print("x2",x2.size())
    #print("y",y.size())
    #print("y2",y2.size())
    """
    Computes pairwise squared l2 distances between tensors x and y.
    Args:
    x: [n, feature_dim].
    y: [m, feature_dim].
    Returns:
    d: [n, m].
    """
    xs = x2
    ys = y2

    xs = xs.unsqueeze(1)
    ys = ys.unsqueeze(0)
    d = xs + ys - 2. * torch.matmul(x, torch.t(y))
    return d

##################
def _flattened_pairwise_distances(reference_embeddings, ref_square, query_embeddings, query_square):
    """
    Calculates flattened tensor of pairwise distances between ref and query.
    Args:
        reference_embeddings: [..., embedding_dim],
          the embedding vectors for the reference frame
        query_embeddings: [..., embedding_dim], 
          the embedding vectors for the query frames.
    Returns:
        dists: [reference_embeddings.size / embedding_dim, query_embeddings.size / embedding_dim]
    """
    dists = _pairwise_distances(query_embeddings, query_square, reference_embeddings, ref_square)
    return dists

def _nn_features_per_object_for_chunk(
    reference_embeddings, ref_square, query_embeddings, query_square, wrong_label_mask):
    """Extracts features for each object using nearest neighbor attention.
    Args:
        reference_embeddings: [n_chunk, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [m_chunk, embedding_dim],
          the embedding vectors for the query frames.
        wrong_label_mask: [n_objects, n_chunk],
          the mask for pixels not used for matching.
    Returns:
        nn_features: A float32 tensor of nearest neighbor features of shape
          [m_chunk, n_objects, n_chunk].
    """
    if reference_embeddings.dtype == torch.float16:
        wrong_label_mask = wrong_label_mask.half()
    else:
        wrong_label_mask = wrong_label_mask.float()

    reference_embeddings_key = reference_embeddings
    query_embeddings_key = query_embeddings
    dists = _flattened_pairwise_distances(reference_embeddings_key, ref_square, query_embeddings_key, query_square)
    
    dists = (torch.unsqueeze(dists, 1) +
            torch.unsqueeze(wrong_label_mask, 0) *
           WRONG_LABEL_PADDING_DISTANCE)
    
    features, _ = torch.min(dists, 2, keepdim=True)
    return features


def _nn_features_per_object_for_chunk_cluster(
    reference_embeddings, ref_square, query_embeddings, query_square):
    """Extracts features for each object using nearest neighbor attention.
    Args:
        reference_embeddings: [n_chunk, embedding_dim,obj_num],
          the embedding vectors for the reference frame.
        query_embeddings: [m_chunk, embedding_dim],
          the embedding vectors for the query frames.
        wrong_label_mask: [n_objects, n_chunk],
          the mask for pixels not used for matching.
    Returns:
        nn_features: A float32 tensor of nearest neighbor features of shape
          [m_chunk, n_objects, n_chunk].
    """
    #print("reference_embeddings.size()",reference_embeddings.size())
    #print("ref_square.size()",ref_square.size())
    #print("query_embeddings.size()",query_embeddings.size())
    #print("query_square.size()",query_square.size())

    dists = _flattened_pairwise_distances(reference_embeddings, ref_square, query_embeddings, query_square)
    """
    dists = (torch.unsqueeze(dists, 1) +
            torch.unsqueeze(wrong_label_mask, 0) *
           WRONG_LABEL_PADDING_DISTANCE)
    """
    dists = torch.unsqueeze(dists, 1)
    features, _ = torch.min(dists, 2, keepdim=True)
    return features

def _nn_features_per_object_for_chunk_proxy(
    reference_embeddings, ref_square, query_embeddings, query_square):
    """Extracts features for each object using nearest neighbor attention.
    Args:
        reference_embeddings: [n_chunk, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [m_chunk, embedding_dim],
          the embedding vectors for the query frames.
        wrong_label_mask: [n_objects, n_chunk],
          the mask for pixels not used for matching.
    Returns:
        nn_features: A float32 tensor of nearest neighbor features of shape
          [m_chunk, n_objects, n_chunk].
    """

    #reference_embeddings_key = reference_embeddings
    #query_embeddings_key = query_embeddings
    dists = _flattened_pairwise_distances(reference_embeddings, ref_square, query_embeddings, query_square)
    #print("dists",dists.size())
    #dists = (torch.unsqueeze(dists, 1))
    
    #features, _ = torch.min(dists, 2, keepdim=True)
    #print("minfeatures",features.size())
    return dists


def _nearest_neighbor_features_per_object_in_chunks_proxy(
    reference_embeddings_flat, query_embeddings_flat, n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    ref_square = reference_embeddings_flat.pow(2).sum(1) #.unsqueeze(0) #
    #reference_embeddings_flat = reference_embeddings_flat.unsqueeze(0)
    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("ref_square.size()",ref_square.size())

    query_square = query_embeddings_flat.pow(2).sum(1)

    all_features = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        features = _nn_features_per_object_for_chunk_proxy(
            reference_embeddings_flat, ref_square, query_embeddings_flat_chunk, query_square_chunk)
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = torch.cat(all_features, dim=0)
    

    return nn_features


def _nearest_neighbor_features_per_object_in_chunks(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    wrong_label_mask = reference_labels_flat < 0.1
    wrong_label_mask = wrong_label_mask.permute(1, 0)
    ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)



    all_features = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        features = _nn_features_per_object_for_chunk(
            reference_embeddings_flat, ref_square, query_embeddings_flat_chunk, query_square_chunk,
            wrong_label_mask)
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = torch.cat(all_features, dim=0)
    

    return nn_features

def _nearest_neighbor_features_per_object_in_chunks_cluster(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    for i in range(right_label_mask.size()[0]):
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(right_label_mask[i]).squeeze(1))

        #print("reference_embeddings_flat_cur.size()[0]",reference_embeddings_flat_cur.size()[0])
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()

        ## pixel < cluster_num, use all pixels
        cluster_num = min(16,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=20)
            #t1 = time.time()
            #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))

            #counts = np.bincount(label)
            label_org = label
            label = torch.from_numpy(label).to(reference_embeddings_flat.device)
            #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
            centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)
            #print("label",label.size())
            #print("centroid",centroid.size())
            #print("counts",counts)
            """
            for j in range(cluster_num):
                reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1))
                reference_embeddings_flat_cur_avg =torch.sum(reference_embeddings_flat_cur,0)/counts[j]
                print("reference_embeddings_flat_cur_avg",reference_embeddings_flat_cur_avg)
                print("centroid[j]",centroid[j])
            """
            #print("len(counts)",len(counts))
            #print("cluster_num",cluster_num)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
            centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
            #print("centroid_avg.size()",centroid_avg.size())
            #print("centroid.size()",centroid.size())
            centroid_all = torch.cat([centroid,centroid_avg],0)
            #print("centroid_all.size()",centroid_all.size())
            #centroid_all_objs.append(centroid_all.unsqueeze(2))
            #centroid_all_objs_squ.append(centroid_all.pow(2).sum(1).unsqueeze(1))
            centroid_all_objs.append(centroid_all)
            centroid_all_objs_squ.append(centroid_all.pow(2).sum(1))
    #centroid_all_objs = torch.cat(centroid_all_objs,2).permute(2,0,1)
    #ref_square = torch.cat(centroid_all_objs_squ,1).permute(1,0)
    #print("centroid_all_objs.size()",centroid_all_objs.size())
    #print("ref_square.size()",ref_square.size())
    all_features = []

    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        features_multi=[]
        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                features_multi.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
            else:
                features = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i], centroid_all_objs_squ[i], query_embeddings_flat_chunk, query_square_chunk)
                features_multi.append(features)
            #print("features.size()",features.size())
        features_multi = torch.cat(features_multi,1).to(reference_embeddings_flat.device)
        #print("features_multi.size()",features_multi.size())
        all_features.append(features_multi)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = torch.cat(all_features, dim=0)
    
    #print("nn_features.size()",nn_features.size())
    return nn_features

def _nearest_neighbor_features_per_object_in_chunks_cluster3(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks, cluster_number):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    right_label_mask_opt = torch.zeros_like(right_label_mask.int())
    reference_embeddings_flat_opt = reference_embeddings_flat+1
    for i in range(right_label_mask.size()[0]):
        index_choice = torch.nonzero(right_label_mask[i]).squeeze(1)
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,index_choice)
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()
        """
        final_mask = torch.tensor([1,0,1,1])
        index_choice=(torch.nonzero(final_mask).squeeze(1))
        cluster_label = torch.tensor([1,2,3])
        for i in range(cluster_label.size(0)):
          final_mask[index_choice[i]] = cluster_label[i]  
        """
        ## pixel < cluster_num, use all pixels
        cluster_num = min(cluster_number,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=50)
            #t1 = time.time()
            #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))
            #print("label.shape",label.shape)
            #print("label",label)
            for j in range(index_choice.size(0)):
                right_label_mask_opt[i][index_choice[j]] = label[j]  +2
            
            #counts = np.bincount(label)
            label_org = label
            label = torch.from_numpy(label).to(reference_embeddings_flat.device)
            #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
            centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)
            """
            label torch.Size([36066])
            centroid torch.Size([16, 100])
            """
            """
            for j in range(cluster_num):
                reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1))
                reference_embeddings_flat_cur_avg =torch.sum(reference_embeddings_flat_cur,0)/counts[j]
                print("reference_embeddings_flat_cur_avg",reference_embeddings_flat_cur_avg)
                print("centroid[j]",centroid[j])
            """
            #print("len(counts)",len(counts))
            #print("cluster_num",cluster_num)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
            #torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))
            #print("torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1).size()",torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1).size())
            #torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)
            #reference_embeddings_flat_opt = reference_embeddings_flat+1

            centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
            centroid_all_objs.append([centroid,centroid_avg])
            centroid_all_objs_squ.append([centroid.pow(2).sum(1),centroid_avg.pow(2).sum(1)])
            #seg_mask  =torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)
            #print("seg_mask.size()",seg_mask.size())
            #reference_embeddings_flat_opt = reference_embeddings_flat_opt * (1-seg_mask)+ centroid_avg * seg_mask
    #print("right_label_mask_opt.size()",right_label_mask_opt.size())
    #print(right_label_mask_opt)
    all_features1 = []
    all_features2 = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        features_multi1=[]
        features_multi2=[]
        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                features_multi1.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
                features_multi2.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
            else:
                features1 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)
                features2 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][1], centroid_all_objs_squ[i][1], query_embeddings_flat_chunk, query_square_chunk)
                features_multi1.append(features1)
                features_multi2.append(features2)
            #print("features.size()",features.size())
        features_multi1 = torch.cat(features_multi1,1).to(reference_embeddings_flat.device)
        features_multi2 = torch.cat(features_multi2,1).to(reference_embeddings_flat.device)
        #print("features_multi.size()",features_multi.size())
        all_features1.append(features_multi1)
        all_features2.append(features_multi2)
    if n_chunks == 1:
        nn_features = [all_features1[0],all_features2[0]]
    else:
        nn_features = [torch.cat(all_features1, dim=0),torch.cat(all_features2, dim=0)]
    
    #print("nn_features.size()",nn_features.size())
    return nn_features,right_label_mask_opt#,reference_embeddings_flat_opt

def _nearest_neighbor_features_per_object_in_chunks_cluster2(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks,cluster_num=16):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    right_label_mask_opt = torch.zeros_like(right_label_mask.int())
    #print("right_label_mask_opt_pre",right_label_mask_opt.size())
    for i in range(right_label_mask.size()[0]):
        index_choice = torch.nonzero(right_label_mask[i]).squeeze(1)
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,index_choice)
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()
        """
        final_mask = torch.tensor([1,0,1,1])
        index_choice=(torch.nonzero(final_mask).squeeze(1))
        cluster_label = torch.tensor([1,2,3])
        for i in range(cluster_label.size(0)):
          final_mask[index_choice[i]] = cluster_label[i]  
        """
        ## pixel < cluster_num, use all pixels
        cluster_num = min(cluster_num,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            try:
                centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=20)
                #t1 = time.time()
                #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))
                #print("label.shape",label.shape)
                #print("label",label)
                #for j in range(index_choice.size(0)):
                #    right_label_mask_opt[i][index_choice[j]] = label[j]  
            
                #counts = np.bincount(label)
                label_org = label
                label = torch.from_numpy(label).to(reference_embeddings_flat.device)
                #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
                centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)
                """
                label torch.Size([36066])
                centroid torch.Size([16, 100])
                """
                """
                for j in range(cluster_num):
                    reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1))
                    reference_embeddings_flat_cur_avg =torch.sum(reference_embeddings_flat_cur,0)/counts[j]
                    print("reference_embeddings_flat_cur_avg",reference_embeddings_flat_cur_avg)
                    print("centroid[j]",centroid[j])
                """
                #print("len(counts)",len(counts))
                #print("cluster_num",cluster_num)
                #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
                centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
                centroid_all_objs.append([centroid,centroid_avg])
                centroid_all_objs_squ.append([centroid.pow(2).sum(1),centroid_avg.pow(2).sum(1)])
            except:
                print("CLUSTERING ERROR!")
                centroid_all_objs.append(None)
                centroid_all_objs_squ.append(None)

    #print("right_label_mask_opt.size()",right_label_mask_opt.size())

    #print(right_label_mask_opt)
    all_features1 = []
    all_features2 = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        features_multi1=[]
        features_multi2=[]
        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                features_multi1.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
                features_multi2.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
            else:
                features1 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)
                features2 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][1], centroid_all_objs_squ[i][1], query_embeddings_flat_chunk, query_square_chunk)
                features_multi1.append(features1)
                features_multi2.append(features2)
            #print("features.size()",features.size())
        features_multi1 = torch.cat(features_multi1,1).to(reference_embeddings_flat.device)
        features_multi2 = torch.cat(features_multi2,1).to(reference_embeddings_flat.device)
        #print("features_multi.size()",features_multi.size())
        all_features1.append(features_multi1)
        all_features2.append(features_multi2)
    if n_chunks == 1:
        nn_features = [all_features1[0],all_features2[0]]
    else:
        nn_features = [torch.cat(all_features1, dim=0),torch.cat(all_features2, dim=0)]
    
    #print("nn_features.size()",nn_features.size())
    return nn_features


def _nearest_neighbor_features_per_object_in_chunks_cluster2_1(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks,cluster_num=16):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    right_label_mask_opt = torch.zeros_like(right_label_mask.int())
    #print("right_label_mask_opt_pre",right_label_mask_opt.size())
    for i in range(right_label_mask.size()[0]):
        index_choice = torch.nonzero(right_label_mask[i]).squeeze(1)
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,index_choice)
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()
        """
        final_mask = torch.tensor([1,0,1,1])
        index_choice=(torch.nonzero(final_mask).squeeze(1))
        cluster_label = torch.tensor([1,2,3])
        for i in range(cluster_label.size(0)):
          final_mask[index_choice[i]] = cluster_label[i]  
        """
        ## pixel < cluster_num, use all pixels
        cluster_num = min(cluster_num,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=20)
            #t1 = time.time()
            #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))
            #print("label.shape",label.shape)
            #print("label",label)
            #for j in range(index_choice.size(0)):
            #    right_label_mask_opt[i][index_choice[j]] = label[j]  
            
            #counts = np.bincount(label)
            label_org = label
            label = torch.from_numpy(label).to(reference_embeddings_flat.device)
            #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
            centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)
            """
            label torch.Size([36066])
            centroid torch.Size([16, 100])
            """
            """
            for j in range(cluster_num):
                reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1))
                reference_embeddings_flat_cur_avg =torch.sum(reference_embeddings_flat_cur,0)/counts[j]
                print("reference_embeddings_flat_cur_avg",reference_embeddings_flat_cur_avg)
                print("centroid[j]",centroid[j])
            """
            #print("len(counts)",len(counts))
            #print("cluster_num",cluster_num)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
            centroid_all_objs.append([centroid])
            centroid_all_objs_squ.append([centroid.pow(2).sum(1)])

    #print("right_label_mask_opt.size()",right_label_mask_opt.size())

    #print(right_label_mask_opt)
    all_features1 = []

    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        features_multi1=[]

        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                features_multi1.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
 
            else:
                features1 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)

                features_multi1.append(features1)

            #print("features.size()",features.size())
        features_multi1 = torch.cat(features_multi1,1).to(reference_embeddings_flat.device)

        #print("features_multi.size()",features_multi.size())
        all_features1.append(features_multi1)

    if n_chunks == 1:
        nn_features = [all_features1[0]]
    else:
        nn_features = [torch.cat(all_features1, dim=0)]
    
    #print("nn_features.size()",nn_features.size())
    return nn_features


def _nearest_neighbor_features_per_object_in_chunks_cluster2_2(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks,cluster_num=16):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    right_label_mask_opt = torch.zeros_like(right_label_mask.int())
    #print("right_label_mask_opt_pre",right_label_mask_opt.size())
    for i in range(right_label_mask.size()[0]):
        index_choice = torch.nonzero(right_label_mask[i]).squeeze(1)
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,index_choice)
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()
        """
        final_mask = torch.tensor([1,0,1,1])
        index_choice=(torch.nonzero(final_mask).squeeze(1))
        cluster_label = torch.tensor([1,2,3])
        for i in range(cluster_label.size(0)):
          final_mask[index_choice[i]] = cluster_label[i]  
        """
        ## pixel < cluster_num, use all pixels
        cluster_num = min(cluster_num,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=20)
            #t1 = time.time()
            #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))
            #print("label.shape",label.shape)
            #print("label",label)
            #for j in range(index_choice.size(0)):
            #    right_label_mask_opt[i][index_choice[j]] = label[j]  
            
            #counts = np.bincount(label)
            label_org = label
            label = torch.from_numpy(label).to(reference_embeddings_flat.device)
            #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
            centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)
            """
            label torch.Size([36066])
            centroid torch.Size([16, 100])
            """
            """
            for j in range(cluster_num):
                reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1))
                reference_embeddings_flat_cur_avg =torch.sum(reference_embeddings_flat_cur,0)/counts[j]
                print("reference_embeddings_flat_cur_avg",reference_embeddings_flat_cur_avg)
                print("centroid[j]",centroid[j])
            """
            #print("len(counts)",len(counts))
            #print("cluster_num",cluster_num)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
            centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
            centroid_all_objs.append([centroid_avg])
            centroid_all_objs_squ.append([centroid_avg.pow(2).sum(1)])

    #print("right_label_mask_opt.size()",right_label_mask_opt.size())

    #print(right_label_mask_opt)
    #all_features1 = []
    all_features2 = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        #features_multi1=[]
        features_multi2=[]
        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                #features_multi1.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
                features_multi2.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
            else:
                #features1 = _nn_features_per_object_for_chunk_cluster(
                #    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)
                features2 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)
                #features_multi1.append(features1)
                features_multi2.append(features2)
            #print("features.size()",features.size())
        #features_multi1 = torch.cat(features_multi1,1).to(reference_embeddings_flat.device)
        features_multi2 = torch.cat(features_multi2,1).to(reference_embeddings_flat.device)
        #print("features_multi.size()",features_multi.size())
        #all_features1.append(features_multi1)
        all_features2.append(features_multi2)
    if n_chunks == 1:
        nn_features = [all_features2[0]]
    else:
        nn_features = [torch.cat(all_features2, dim=0)]
    
    #print("nn_features.size()",nn_features.size())
    return nn_features

def _nearest_neighbor_features_per_object_in_chunks_cluster2_stronger(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks,cluster_num=16):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    right_label_mask_opt = torch.zeros_like(right_label_mask.int())
    #print("right_label_mask_opt_pre",right_label_mask_opt.size())
    for i in range(right_label_mask.size()[0]):
        index_choice = torch.nonzero(right_label_mask[i]).squeeze(1)
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,index_choice)
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()
        """
        final_mask = torch.tensor([1,0,1,1])
        index_choice=(torch.nonzero(final_mask).squeeze(1))
        cluster_label = torch.tensor([1,2,3])
        for i in range(cluster_label.size(0)):
          final_mask[index_choice[i]] = cluster_label[i]  
        """
        ## pixel < cluster_num, use all pixels
        cluster_num = min(cluster_num,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=100)
            #t1 = time.time()
            #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))
            #print("label.shape",label.shape)
            #print("label",label)
            #for j in range(index_choice.size(0)):
            #    right_label_mask_opt[i][index_choice[j]] = label[j]  
            
            #counts = np.bincount(label)
            label_org = label
            label = torch.from_numpy(label).to(reference_embeddings_flat.device)
            #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
            centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)
            """
            label torch.Size([36066])
            centroid torch.Size([16, 100])
            """
            """
            for j in range(cluster_num):
                reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1))
                reference_embeddings_flat_cur_avg =torch.sum(reference_embeddings_flat_cur,0)/counts[j]
                print("reference_embeddings_flat_cur_avg",reference_embeddings_flat_cur_avg)
                print("centroid[j]",centroid[j])
            """
            #print("len(counts)",len(counts))
            #print("cluster_num",cluster_num)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
            centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
            centroid_all_objs.append([centroid,centroid_avg])
            centroid_all_objs_squ.append([centroid.pow(2).sum(1),centroid_avg.pow(2).sum(1)])

    #print("right_label_mask_opt.size()",right_label_mask_opt.size())

    #print(right_label_mask_opt)
    all_features1 = []
    all_features2 = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        features_multi1=[]
        features_multi2=[]
        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                features_multi1.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
                features_multi2.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
            else:
                features1 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)
                features2 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][1], centroid_all_objs_squ[i][1], query_embeddings_flat_chunk, query_square_chunk)
                features_multi1.append(features1)
                features_multi2.append(features2)
            #print("features.size()",features.size())
        features_multi1 = torch.cat(features_multi1,1).to(reference_embeddings_flat.device)
        features_multi2 = torch.cat(features_multi2,1).to(reference_embeddings_flat.device)
        #print("features_multi.size()",features_multi.size())
        all_features1.append(features_multi1)
        all_features2.append(features_multi2)
    if n_chunks == 1:
        nn_features = [all_features1[0],all_features2[0]]
    else:
        nn_features = [torch.cat(all_features1, dim=0),torch.cat(all_features2, dim=0)]
    
    #print("nn_features.size()",nn_features.size())
    return nn_features



def _nearest_neighbor_features_per_object_in_chunks_cluster2_random_shuffle(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks,cluster_num=16):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    #ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    #kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    #reference_embeddings_flat_cluster_label = kmeans.fit_predict(reference_embeddings_flat)
    #right_label_mask torch.Size([ 13689,2])
    #reference_embeddings_flat torch.Size([13689, 100])

    right_label_mask = reference_labels_flat > 0.9
    right_label_mask = right_label_mask.permute(1, 0)
    #print("right_label_mask",right_label_mask.size())
    #print("reference_embeddings_flat",reference_embeddings_flat.size())
    
    centroid_all_objs=[]
    centroid_all_objs_squ=[]
    right_label_mask_opt = torch.zeros_like(right_label_mask.int())
    #print("right_label_mask_opt_pre",right_label_mask_opt.size())
    for i in range(right_label_mask.size()[0]):
        index_choice = torch.nonzero(right_label_mask[i]).squeeze(1)
        reference_embeddings_flat_cur = torch.index_select(reference_embeddings_flat,0,index_choice)
        reference_embeddings_flat_cur_np  = reference_embeddings_flat_cur.detach().cpu().numpy()
        # start clustering
        #t0 = time.time()
        """
        final_mask = torch.tensor([1,0,1,1])
        index_choice=(torch.nonzero(final_mask).squeeze(1))
        cluster_label = torch.tensor([1,2,3])
        for i in range(cluster_label.size(0)):
          final_mask[index_choice[i]] = cluster_label[i]  
        """
        ## pixel < cluster_num, use all pixels
        cluster_num = min(cluster_num,reference_embeddings_flat_cur.size()[0])
        if cluster_num==0:
            centroid_all_objs.append(None)
            centroid_all_objs_squ.append(None)
        else:
            centroid, label = kmeans2(reference_embeddings_flat_cur_np, cluster_num, minit='points',iter=20)
            #t1 = time.time()
            #print ("Total time running %s: %s seconds" %("cluster", str(t1-t0)))
            #print("label.shape",label.shape)
            #print("label",label)
            #for j in range(index_choice.size(0)):
            #    right_label_mask_opt[i][index_choice[j]] = label[j]  
            
            #counts = np.bincount(label)

            random.shuffle(label)
            label_org = label
            label = torch.from_numpy(label).to(reference_embeddings_flat.device)
            #counts = torch.from_numpy(counts).to(reference_embeddings_flat.device)
            centroid = torch.from_numpy(centroid).to(reference_embeddings_flat.device)

            #print("len(counts)",len(counts))
            #print("cluster_num",cluster_num)
            #centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/counts[j]).unsqueeze(0) for j in range(cluster_num)],0)
            centroid_avg = torch.cat([(torch.sum(torch.index_select(reference_embeddings_flat,0,torch.nonzero(torch.tensor((label==j))).to(reference_embeddings_flat.device).squeeze(1)),0)/(np.sum(label_org==j))).unsqueeze(0) for j in np.unique(label_org)],0)
            centroid_all_objs.append([centroid_avg,centroid_avg])
            centroid_all_objs_squ.append([centroid_avg.pow(2).sum(1),centroid_avg.pow(2).sum(1)])

    #print("right_label_mask_opt.size()",right_label_mask_opt.size())

    #print(right_label_mask_opt)
    all_features1 = []
    all_features2 = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        #print("query_square_chunk.size()",query_square_chunk.size())
        features_multi1=[]
        features_multi2=[]
        for i in range(right_label_mask.size()[0]):
            if centroid_all_objs[i] == None:
                features_multi1.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
                features_multi2.append(torch.ones((query_embeddings_flat_chunk.size()[0],1,1)).to(reference_embeddings_flat.device)* WRONG_LABEL_PADDING_DISTANCE)
            else:
                features1 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][0], centroid_all_objs_squ[i][0], query_embeddings_flat_chunk, query_square_chunk)
                features2 = _nn_features_per_object_for_chunk_cluster(
                    centroid_all_objs[i][1], centroid_all_objs_squ[i][1], query_embeddings_flat_chunk, query_square_chunk)
                features_multi1.append(features1)
                features_multi2.append(features2)
            #print("features.size()",features.size())
        features_multi1 = torch.cat(features_multi1,1).to(reference_embeddings_flat.device)
        features_multi2 = torch.cat(features_multi2,1).to(reference_embeddings_flat.device)
        #print("features_multi.size()",features_multi.size())
        all_features1.append(features_multi1)
        all_features2.append(features_multi2)
    if n_chunks == 1:
        nn_features = [all_features1[0],all_features2[0]]
    else:
        nn_features = [torch.cat(all_features1, dim=0),torch.cat(all_features2, dim=0)]
    
    #print("nn_features.size()",nn_features.size())
    return nn_features


def global_matching_proxy(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """
    
    #assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])

    #print("global_matching_proxy")
    #print("reference_embeddings",reference_embeddings.size())
    #print("query_embeddings",query_embeddings.size())
    #print("reference_labels",reference_labels.size())
    ##
    # reference_embeddings torch.Size([3, 100])
    # query_embeddings torch.Size([117, 117, 100])
    # reference_labels torch.Size([117, 117, 3])

    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = reference_labels.size(2)

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
        selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                               (w + w_pad) // atrous_rate, atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]
        is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
        reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points


    reference_labels_flat = reference_labels.view(-1, obj_nums)

    #reference_embeddings = reference_embeddings.permute(1,0)
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat,
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)

    nn_features = _nearest_neighbor_features_per_object_in_chunks_proxy(
        reference_embeddings, query_embeddings_flat,
        n_chunks)

    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape



def global_matching_cluster(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """
    
    assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])
    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = reference_labels.size(2)

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
        selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                               (w + w_pad) // atrous_rate, atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]
        is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
        reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

    reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
    reference_labels_flat = reference_labels.view(-1, obj_nums)
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat,
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)



    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("query_embeddings_flat.size()",query_embeddings_flat.size())
    #print("reference_labels_flat.size()",reference_labels_flat.size())


    nn_features = _nearest_neighbor_features_per_object_in_chunks_cluster(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)

    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape


def global_matching_cluster2(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """
    
    assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])
    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = reference_labels.size(2)

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
        selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                               (w + w_pad) // atrous_rate, atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]
        is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
        reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

    reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
    reference_labels_flat = reference_labels.view(-1, obj_nums)
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat,
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 2, device=all_ref_fg.device)

    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)



    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("query_embeddings_flat.size()",query_embeddings_flat.size())
    #print("reference_labels_flat.size()",reference_labels_flat.size())

    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]
    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post


def global_matching_cluster2_1(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0,cluster_num=16):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """
    
    assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])
    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = reference_labels.size(2)

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
        selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                               (w + w_pad) // atrous_rate, atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]
        is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
        reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

    reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
    reference_labels_flat = reference_labels.view(-1, obj_nums)
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat,
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 2, device=all_ref_fg.device)

    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)



    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("query_embeddings_flat.size()",query_embeddings_flat.size())
    #print("reference_labels_flat.size()",reference_labels_flat.size())

    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2_1(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks,cluster_num)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]
    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post

def global_matching_cluster2_2(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0,cluster_num=16):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """
    
    assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])
    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = reference_labels.size(2)

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
        selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                               (w + w_pad) // atrous_rate, atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]
        is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
        reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

    reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
    reference_labels_flat = reference_labels.view(-1, obj_nums)
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat,
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 2, device=all_ref_fg.device)

    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)



    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("query_embeddings_flat.size()",query_embeddings_flat.size())
    #print("reference_labels_flat.size()",reference_labels_flat.size())

    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2_2(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks,cluster_num)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]
    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post
# TODO
def global_matching_for_eval_cluster(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    ref_num = len(all_reference_labels)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    

    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]
    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post


# TODO
def global_matching_for_eval_cluster2(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0,cluster_number=16):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    
    ref_num = len(all_reference_labels)
    #print("ref_num",ref_num)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    

    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    #print("reference_labels_flat.size() pre",reference_labels_flat.size())
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat_opt = torch.zeros((obj_nums,h*w*ref_num)).int().to(reference_embeddings_flat.device)
    reference_labels_flat_index_choice=torch.nonzero(reference_labels_flat)
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    #print("reference_labels_flat_opt.size()",reference_labels_flat_opt.size())    
    #print("reference_labels_flat_index_choice.size()",reference_labels_flat_index_choice.size())
    #print("reference_labels_flat_index_choice",reference_labels_flat_index_choice)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device),None#,None
    #print("reference_labels_flat.size() post",reference_labels_flat.size())
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features_list,cluster_label_map = _nearest_neighbor_features_per_object_in_chunks_cluster3(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks,cluster_number) #,reference_embeddings_flat_opt
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]
    print(ref_num*obj_nums,"_",h,"_", w)
    for i in range(cluster_label_map.size(0)):
        for j in range(cluster_label_map.size(1)):
            idx_trans = reference_labels_flat_index_choice[j][0]
            cur_cluster_label_map = cluster_label_map[i][j]
            reference_labels_flat_opt[i][idx_trans] =   cur_cluster_label_map
    reference_labels_flat_opt = reference_labels_flat_opt.view(-1,h, w)
    #reference_embeddings_flat_opt = reference_embeddings_flat_opt.view(-1,h,w)
    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post,reference_labels_flat_opt#,reference_embeddings_flat_opt


# TODO
def global_matching_for_eval_cluster2_cp(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0,cluster_number=16):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    
    ref_num = len(all_reference_labels)
    #print("ref_num",ref_num)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    

    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    #print("reference_labels_flat.size() pre",reference_labels_flat.size())
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat_opt = torch.zeros((obj_nums,h*w*ref_num)).int().to(reference_embeddings_flat.device)
    reference_labels_flat_index_choice=torch.nonzero(reference_labels_flat)
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    #print("reference_labels_flat_opt.size()",reference_labels_flat_opt.size())    
    #print("reference_labels_flat_index_choice.size()",reference_labels_flat_index_choice.size())
    #print("reference_labels_flat_index_choice",reference_labels_flat_index_choice)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    #print("reference_labels_flat.size() post",reference_labels_flat.size())
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks,cluster_number)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]


    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post

# TODO
def global_matching_for_eval_cluster2_cp_stronger(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0,cluster_number=16):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    
    ref_num = len(all_reference_labels)
    #print("ref_num",ref_num)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    

    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    #print("reference_labels_flat.size() pre",reference_labels_flat.size())
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat_opt = torch.zeros((obj_nums,h*w*ref_num)).int().to(reference_embeddings_flat.device)
    reference_labels_flat_index_choice=torch.nonzero(reference_labels_flat)
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    #print("reference_labels_flat_opt.size()",reference_labels_flat_opt.size())    
    #print("reference_labels_flat_index_choice.size()",reference_labels_flat_index_choice.size())
    #print("reference_labels_flat_index_choice",reference_labels_flat_index_choice)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    #print("reference_labels_flat.size() post",reference_labels_flat.size())
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2_stronger(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks,cluster_number)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]


    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post

# TODO
def global_matching_for_eval_cluster2_random_shuffle(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0,cluster_number=16):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    
    ref_num = len(all_reference_labels)
    #print("ref_num",ref_num)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    

    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    #print("reference_labels_flat.size() pre",reference_labels_flat.size())
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat_opt = torch.zeros((obj_nums,h*w*ref_num)).int().to(reference_embeddings_flat.device)
    reference_labels_flat_index_choice=torch.nonzero(reference_labels_flat)
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    #print("reference_labels_flat_opt.size()",reference_labels_flat_opt.size())    
    #print("reference_labels_flat_index_choice.size()",reference_labels_flat_index_choice.size())
    #print("reference_labels_flat_index_choice",reference_labels_flat_index_choice)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    #print("reference_labels_flat.size() post",reference_labels_flat.size())
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features_list = _nearest_neighbor_features_per_object_in_chunks_cluster2_random_shuffle(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks,cluster_number)
    #print("global_matching_proxy_nn_features",nn_features.size())
    nn_features_reshape_post=[]


    for nn_features in nn_features_list:
        nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
        nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

        if ori_size is not None:
            nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
            nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
                mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

        if use_float16:
            nn_features_reshape = nn_features_reshape.float()
        nn_features_reshape_post.append(nn_features_reshape)
    nn_features_reshape_post = torch.cat(nn_features_reshape_post,4)
    return nn_features_reshape_post


def global_matching(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """
    
    assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])
    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = reference_labels.size(2)

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
        selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                               (w + w_pad) // atrous_rate, atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]
        is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
        reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

    reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
    reference_labels_flat = reference_labels.view(-1, obj_nums)
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat,
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)



    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("query_embeddings_flat.size()",query_embeddings_flat.size())
    #print("reference_labels_flat.size()",reference_labels_flat.size())


    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)

    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape


def global_matching_for_eval(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    ref_num = len(all_reference_labels)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    

    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks)

    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape



# TODO
def global_matching_for_eval_proxy(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True, atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    #print("type(all_reference_embeddings)",type(all_reference_embeddings))
    #print("all_reference_embeddings.size()",all_reference_embeddings.size())

    h, w, embedding_dim = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    ref_num = len(all_reference_labels)
    n_chunks *= ref_num
    """
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = torch.zeros(h + h_pad, w + w_pad, device=query_embeddings.device)
            selected_points = selected_points.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                   (w + w_pad) // atrous_rate, atrous_rate)
            selected_points[:, 0, :, 0] = 1.
            selected_points = selected_points.view(h + h_pad, w + w_pad, 1)[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = reference_labels.sum(dim=(0, 1)) > (atrous_obj_pixel_num * atrous_rate ** 2)
                reference_labels[:, :, is_big_obj] = reference_labels[:, :, is_big_obj] * selected_points

            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)
            
        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)
        else:

            for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0  or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                        reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                    reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                     (w + w_pad) // atrous_rate, atrous_rate, -1)
                    reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                    reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
            

                reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
                reference_labels_flat = reference_labels.view(-1, obj_nums)

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
            reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    
    """
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    
    #all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    #reference_labels_flat = torch.masked_select(reference_labels_flat, 
    #    all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if ref_num == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    #reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
    #    all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    """
    reference_embeddings_flat.size() torch.Size([1, 100])
    query_embeddings_flat.size() torch.Size([37845, 100])
    nn_features.size() torch.Size([37845, 1])
    Ref Frame: 00000.jpg, Time: 0.42488956451416016
    reference_embeddings_flat.size() torch.Size([1, 100])
    query_embeddings_flat.size() torch.Size([37845, 100])
    nn_features.size() torch.Size([37845, 1])

    """
    #print("reference_embeddings_flat.size()",reference_embeddings_flat.size())
    #print("query_embeddings_flat.size()",query_embeddings_flat.size())
    nn_features = _nearest_neighbor_features_per_object_in_chunks_proxy(
        all_reference_embeddings, query_embeddings_flat, n_chunks)
    #print("nn_features.size()",nn_features.size())
    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape
########################################################################LOCAL_DIST_MAP
def local_pairwise_distances(
    x, y, max_distance=9, atrous_rate=1, allow_downsample=False):
    """Computes pairwise squared l2 distances using a local search window.
        Use for-loop for saving memory.
    Args:
        x: Float32 tensor of shape [height, width, feature_dim].
        y: Float32 tensor of shape [height, width, feature_dim].
        max_distance: Integer, the maximum distance in pixel coordinates
          per dimension which is considered to be in the search window.
        atrous_rate: Integer, the atrous rate of local matching.
        allow_downsample: Bool, if "True", downsample x and y
          with a stride of 2.
    Returns:
        Float32 distances tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """
    if allow_downsample:
        ori_height, ori_width, _ = x.size()
        x = x.permute(2, 0, 1).unsqueeze(0)
        y = y.permute(2, 0, 1).unsqueeze(0)
        down_size = (int(ori_height/2) + 1, int(ori_width/2) + 1)
        x = F.interpolate(x, size=down_size, mode='bilinear', align_corners=True)
        y = F.interpolate(y, size=down_size, mode='bilinear', align_corners=True)
        x = x.squeeze(0).permute(1, 2, 0)
        y = y.squeeze(0).permute(1, 2, 0)

    pad_max_distance = max_distance - max_distance % atrous_rate
    padded_y =nn.functional.pad(y, 
        (0, 0, pad_max_distance, pad_max_distance, pad_max_distance, pad_max_distance), 
        mode='constant', value=WRONG_LABEL_PADDING_DISTANCE)

    height, width, _ = x.size()
    dists = []
    for y in range(2 * pad_max_distance // atrous_rate + 1):
        y_start = y * atrous_rate
        y_end = y_start + height
        y_slice = padded_y[y_start:y_end]
        for x in range(2 * max_distance + 1):
            x_start = x * atrous_rate
            x_end = x_start + width
            offset_y = y_slice[:, x_start:x_end]
            dist = torch.sum(torch.pow((x-offset_y),2), dim=2)
            dists.append(dist)
    dists = torch.stack(dists, dim=2)

    return dists

def local_pairwise_distances_parallel(
    x, y, max_distance=9, atrous_rate=1, allow_downsample=True):
    """Computes pairwise squared l2 distances using a local search window.
    Args:
        x: Float32 tensor of shape [height, width, feature_dim].
        y: Float32 tensor of shape [height, width, feature_dim].
        max_distance: Integer, the maximum distance in pixel coordinates
          per dimension which is considered to be in the search window.
        atrous_rate: Integer, the atrous rate of local matching.
        allow_downsample: Bool, if "True", downsample x and y
          with a stride of 2.
    Returns:
        Float32 distances tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """ 
    ori_height, ori_width, _ = x.size()
    x = x.permute(2, 0, 1).unsqueeze(0)
    y = y.permute(2, 0, 1).unsqueeze(0)
    if allow_downsample:
        down_size = (int(ori_height/2) + 1, int(ori_width/2) + 1)
        x = F.interpolate(x, size=down_size, mode='bilinear', align_corners=True)
        y = F.interpolate(y, size=down_size, mode='bilinear', align_corners=True)

    _, channels, height, width = x.size()

    x2 = x.pow(2).sum(1).view(height, width, 1)

    y2 = y.pow(2).sum(1).view(1, 1, height, width)

    pad_max_distance = max_distance - max_distance % atrous_rate
    
    padded_y = F.pad(y, (pad_max_distance, pad_max_distance, pad_max_distance, pad_max_distance))
    padded_y2 = F.pad(y2, (pad_max_distance, pad_max_distance, pad_max_distance, pad_max_distance), 
        mode='constant', value=WRONG_LABEL_PADDING_DISTANCE)

    offset_y = F.unfold(padded_y, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(channels, height * width, -1).permute(1, 0, 2)
    offset_y2 = F.unfold(padded_y2, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(height, width, -1)
    x = x.view(channels, height * width, -1).permute(1, 2, 0)

    dists = x2 + offset_y2 - 2. * torch.matmul(x, offset_y).view(height, width, -1)
    
    return dists




def local_matching(
    prev_frame_embedding, query_embedding, prev_frame_labels,
    dis_bias=0., multi_local_distance=[15], 
    ori_size=None, atrous_rate=1, use_float16=True, allow_downsample=True, allow_parallel=True):
    """Computes nearest neighbor features while only allowing local matches.
    Args:
        prev_frame_embedding: [height, width, embedding_dim],
          the embedding vectors for the last frame.
        query_embedding: [height, width, embedding_dim],
          the embedding vectors for the query frames.
        prev_frame_labels: [height, width, n_objects], 
        the class labels of the previous frame.
        multi_local_distance: A list of Integer, 
          a list of maximum distance allowed for local matching.
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of local matching.
        use_float16: Bool, if "True", use float16 type for matching.
        allow_downsample: Bool, if "True", downsample prev_frame_embedding and query_embedding
          with a stride of 2.
        allow_parallel: Bool, if "True", do matching in a parallel way. If "False", do matching in
          a for-loop way, which will save GPU memory.
    Returns:
        nn_features: A float32 np.array of nearest neighbor features of shape
          [1, height, width, n_objects, 1].
    """
    max_distance = multi_local_distance[-1]

    if ori_size is None:
        height, width = prev_frame_embedding.size()[:2]
        ori_size = (height, width)

    obj_num = prev_frame_labels.size(2)
    pad = torch.ones(1, device=prev_frame_embedding.device) * WRONG_LABEL_PADDING_DISTANCE
    if use_float16:
        query_embedding = query_embedding.half()
        prev_frame_embedding = prev_frame_embedding.half()
        pad = pad.half()

    if allow_parallel:
        d = local_pairwise_distances_parallel(query_embedding, prev_frame_embedding, 
            max_distance=max_distance, atrous_rate=atrous_rate, allow_downsample=allow_downsample)
    else:
        d = local_pairwise_distances(query_embedding, prev_frame_embedding, 
        max_distance=max_distance, atrous_rate=atrous_rate, allow_downsample=allow_downsample)
        
    height, width = d.size()[:2]
    
    labels = prev_frame_labels.permute(2, 0, 1).unsqueeze(1)
    if (height, width) != ori_size:
        labels = F.interpolate(labels, size=(height, width), mode='nearest')

    pad_max_distance = max_distance - max_distance % atrous_rate
    atrous_max_distance = pad_max_distance // atrous_rate

    padded_labels = F.pad(labels,
                        (pad_max_distance, pad_max_distance,
                         pad_max_distance, pad_max_distance,
                         ), mode='constant', value=0)
    offset_masks = F.unfold(padded_labels, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(obj_num, height, width, -1).permute(1, 2, 3, 0) > 0.9
    
    d_tiled = d.unsqueeze(-1).expand((-1,-1,-1,obj_num))  # h, w, num_local_pos, obj_num

    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists, pos = torch.min(d_masked, dim=2)
    multi_dists = [dists.permute(2, 0, 1).unsqueeze(1)]  # n_objects, num_multi_local, h, w
    
    reshaped_d_masked = d_masked.view(height, width, 2 * atrous_max_distance + 1, 
        2 * atrous_max_distance + 1, obj_num)
    for local_dis in multi_local_distance[:-1]:
        local_dis = local_dis // atrous_rate
        start_idx = atrous_max_distance - local_dis
        end_idx = atrous_max_distance + local_dis + 1
        new_d_masked = reshaped_d_masked[:, :, start_idx:end_idx, start_idx:end_idx, :].contiguous()
        new_d_masked = new_d_masked.view(height, width, -1, obj_num)
        new_dists, _ = torch.min(new_d_masked, dim=2)
        new_dists = new_dists.permute(2, 0, 1).unsqueeze(1)
        multi_dists.append(new_dists)

    multi_dists = torch.cat(multi_dists, dim=1)
    multi_dists = (torch.sigmoid(multi_dists + dis_bias.view(-1, 1, 1, 1)) - 0.5) * 2

    if use_float16:
        multi_dists = multi_dists.float()
        
    if (height, width) != ori_size:
        multi_dists = F.interpolate(multi_dists, size=ori_size, 
            mode='bilinear', align_corners=True)
    multi_dists = multi_dists.permute(2, 3, 0, 1)
    multi_dists = multi_dists.view(1, ori_size[0], ori_size[1], obj_num, -1)

    return multi_dists



def local_matching_proxy(
    prev_frame_embedding, query_embedding, prev_frame_labels,
    dis_bias=0., multi_local_distance=[15], 
    ori_size=None, atrous_rate=1, use_float16=True, allow_downsample=True, allow_parallel=True):
    """Computes nearest neighbor features while only allowing local matches.
    Args:
        prev_frame_embedding: [height, width, embedding_dim],
          the embedding vectors for the last frame.
        query_embedding: [height, width, embedding_dim],
          the embedding vectors for the query frames.
        prev_frame_labels: [height, width, n_objects], 
        the class labels of the previous frame.
        multi_local_distance: A list of Integer, 
          a list of maximum distance allowed for local matching.
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of local matching.
        use_float16: Bool, if "True", use float16 type for matching.
        allow_downsample: Bool, if "True", downsample prev_frame_embedding and query_embedding
          with a stride of 2.
        allow_parallel: Bool, if "True", do matching in a parallel way. If "False", do matching in
          a for-loop way, which will save GPU memory.
    Returns:
        nn_features: A float32 np.array of nearest neighbor features of shape
          [1, height, width, n_objects, 1].
    """
    max_distance = multi_local_distance[-1]

    if ori_size is None:
        height, width = prev_frame_embedding.size()[:2]
        ori_size = (height, width)

    obj_num = prev_frame_labels.size(2)
    pad = torch.ones(1, device=prev_frame_embedding.device) * WRONG_LABEL_PADDING_DISTANCE
    if use_float16:
        query_embedding = query_embedding.half()
        prev_frame_embedding = prev_frame_embedding.half()
        pad = pad.half()

    if allow_parallel:
        d = local_pairwise_distances_parallel(query_embedding, prev_frame_embedding, 
            max_distance=max_distance, atrous_rate=atrous_rate, allow_downsample=allow_downsample)
    else:
        d = local_pairwise_distances(query_embedding, prev_frame_embedding, 
        max_distance=max_distance, atrous_rate=atrous_rate, allow_downsample=allow_downsample)
        
    height, width = d.size()[:2]
    
    labels = prev_frame_labels.permute(2, 0, 1).unsqueeze(1)
    if (height, width) != ori_size:
        labels = F.interpolate(labels, size=(height, width), mode='nearest')

    pad_max_distance = max_distance - max_distance % atrous_rate
    atrous_max_distance = pad_max_distance // atrous_rate

    padded_labels = F.pad(labels,
                        (pad_max_distance, pad_max_distance,
                         pad_max_distance, pad_max_distance,
                         ), mode='constant', value=0)
    offset_masks = F.unfold(padded_labels, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(obj_num, height, width, -1).permute(1, 2, 3, 0) > 0.9
    
    d_tiled = d.unsqueeze(-1).expand((-1,-1,-1,obj_num))  # h, w, num_local_pos, obj_num

    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists, pos = torch.min(d_masked, dim=2)
    multi_dists = [dists.permute(2, 0, 1).unsqueeze(1)]  # n_objects, num_multi_local, h, w
    
    reshaped_d_masked = d_masked.view(height, width, 2 * atrous_max_distance + 1, 
        2 * atrous_max_distance + 1, obj_num)
    for local_dis in multi_local_distance[:-1]:
        local_dis = local_dis // atrous_rate
        start_idx = atrous_max_distance - local_dis
        end_idx = atrous_max_distance + local_dis + 1
        new_d_masked = reshaped_d_masked[:, :, start_idx:end_idx, start_idx:end_idx, :].contiguous()
        new_d_masked = new_d_masked.view(height, width, -1, obj_num)
        new_dists, _ = torch.min(new_d_masked, dim=2)
        new_dists = new_dists.permute(2, 0, 1).unsqueeze(1)
        multi_dists.append(new_dists)

    multi_dists = torch.cat(multi_dists, dim=1)
    multi_dists = (torch.sigmoid(multi_dists + dis_bias.view(-1, 1, 1, 1)) - 0.5) * 2

    if use_float16:
        multi_dists = multi_dists.float()
        
    if (height, width) != ori_size:
        multi_dists = F.interpolate(multi_dists, size=ori_size, 
            mode='bilinear', align_corners=True)
    multi_dists = multi_dists.permute(2, 3, 0, 1)
    multi_dists = multi_dists.view(1, ori_size[0], ori_size[1], obj_num, -1)

    return multi_dists



def local_matching_topk(topk_num,
    prev_frame_embedding, query_embedding, prev_frame_labels,
    dis_bias=0., multi_local_distance=[15], 
    ori_size=None, atrous_rate=1, use_float16=True, allow_downsample=True, allow_parallel=True):
    """
    local_matching_input = [prev_frame_embedding, query_embedding, prev_frame_labels,  dis_bias, multi_local_distance,  ori_size, atrous_rate, use_float16, allow_downsample, allow_parallel]
    f = open('local_matching_input.pk1', 'wb')  # 必须以二进制打开，否则有错
    pickle.dump(local_matching_input, f, 0)
    f.close()
    """



    """Computes nearest neighbor features while only allowing local matches.
    Args:
        prev_frame_embedding: [height, width, embedding_dim],
          the embedding vectors for the last frame.
        query_embedding: [height, width, embedding_dim],
          the embedding vectors for the query frames.
        prev_frame_labels: [height, width, n_objects], 
        the class labels of the previous frame.
        multi_local_distance: A list of Integer, 
          a list of maximum distance allowed for local matching.
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of local matching.
        use_float16: Bool, if "True", use float16 type for matching.
        allow_downsample: Bool, if "True", downsample prev_frame_embedding and query_embedding
          with a stride of 2.
        allow_parallel: Bool, if "True", do matching in a parallel way. If "False", do matching in
          a for-loop way, which will save GPU memory.
    Returns:
        nn_features: A float32 np.array of nearest neighbor features of shape
          [1, height, width, n_objects, 1].
    """
    max_distance = multi_local_distance[-1]
    #print("multi_local_distance:",multi_local_distance)
    if ori_size is None:
        height, width = prev_frame_embedding.size()[:2]
        ori_size = (height, width)

    obj_num = prev_frame_labels.size(2)
    pad = torch.ones(1, device=prev_frame_embedding.device) * WRONG_LABEL_PADDING_DISTANCE
    if use_float16:
        query_embedding = query_embedding.half()
        prev_frame_embedding = prev_frame_embedding.half()
        pad = pad.half()

    if allow_parallel:
        d = local_pairwise_distances_parallel(query_embedding, prev_frame_embedding, 
            max_distance=max_distance, atrous_rate=atrous_rate)
    else:
        d = local_pairwise_distances(query_embedding, prev_frame_embedding, 
        max_distance=max_distance, atrous_rate=atrous_rate)
        
    height, width = d.size()[:2]
    labels = prev_frame_labels.permute(2, 0, 1).unsqueeze(1)
    labels = F.interpolate(labels, size=(height, width), mode='nearest')

    pad_max_distance = max_distance - max_distance % atrous_rate
    atrous_max_distance = pad_max_distance // atrous_rate
    padded_labels = F.pad(labels,
                        (pad_max_distance, pad_max_distance,
                         pad_max_distance, pad_max_distance,
                         ), mode='constant', value=0)
    offset_masks = F.unfold(padded_labels, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(obj_num, height, width, -1).permute(1, 2, 3, 0) > 0.9
    d_tiled = d.unsqueeze(-1).expand((-1,-1,-1,obj_num))  # h, w, num_local_pos, obj_num

    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists, pos = torch.min(d_masked, dim=2)
    topk_dists,topk_pos= d_masked.topk(topk_num,dim=2,largest=False,sorted=True) #######
    """
    a=topk_dists[:,:,0,:]
    b=topk_dists[:,:,1,:]
    delta_top2 = b-a
    """

    multi_dists = [topk_dists.permute(3,2,0,1)]  # n_objects, num_multi_local, h, w
    pos_list  = [topk_pos]
    
    reshaped_d_masked = d_masked.view(height, width, 2 * atrous_max_distance + 1, 
        2 * atrous_max_distance + 1, obj_num)
    
    for local_dis in multi_local_distance[:-1]:
        local_dis = local_dis // atrous_rate
        start_idx = atrous_max_distance - local_dis
        end_idx = atrous_max_distance + local_dis + 1
        new_d_masked = reshaped_d_masked[:, :, start_idx:end_idx, start_idx:end_idx, :].contiguous()
        new_d_masked = new_d_masked.view(height, width, -1, obj_num)

        topk_new_dists, topk_new_pos = new_d_masked.topk(topk_num,dim=2,largest=False,sorted=True) ######
        topk_new_dists = topk_new_dists.permute(3,2,0,1)
        pos_list.append(topk_new_pos)
        topk_new_pos = topk_new_pos.permute(3,2,0,1)

        
        multi_dists.append(topk_new_dists)
      


    multi_dists = torch.cat(multi_dists, dim=1)
    multi_dists_org = multi_dists

    multi_dists = (torch.sigmoid(multi_dists + dis_bias.view(-1, 1, 1, 1)) - 0.5) * 2
    #print("multi_dists.size()", multi_dists.size())
    if use_float16:
        multi_dists = multi_dists.float()

    ori_height = ori_size[0]
    ori_width = ori_size[1]
    
    multi_dists_org = F.interpolate(multi_dists_org, size=(ori_height, ori_width), 
        mode='bilinear', align_corners=True)
    multi_dists_org = multi_dists_org.permute(2, 3, 0, 1)
    multi_dists_org = multi_dists_org.view(1, ori_height, ori_width, obj_num, -1)

    multi_dists = F.interpolate(multi_dists, size=(ori_height, ori_width), 
        mode='bilinear', align_corners=True)
    multi_dists = multi_dists.permute(2, 3, 0, 1)
    multi_dists = multi_dists.view(1, ori_height, ori_width, obj_num, -1)

    return multi_dists,multi_dists_org,pos_list,pad_max_distance