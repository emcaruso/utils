import numpy as np
import pickle
import torch
import itertools
import os


def save_memory_mapped_arr(data,folder, name, flat=False, dtype=None):
    if dtype is None:
        dtype = data.dtype
    # Define the path to the memory-mapped binary file
    file_path = folder+"/"+name+'.bin'
    shape = data.shape
    # Create a memory-mapped array and write the data
    mmapped_array = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape)
    mmapped_array[:] = data[:]
    del mmapped_array  # Close the memory-mapped file when done

    if not flat:
        np.save(folder+"/"+name+'.npy', shape)


def load_memory_mapped_arr(folder, name, dtype, flat=False):
    # To read the data from the memory-mapped file:
    npy_path = folder+"/"+name+'.npy'
    retrieved_data = None
    if os.path.exists(npy_path) and not flat: 
        shape = tuple( np.load( npy_path ).tolist() )
        retrieved_data = np.memmap(folder+"/"+name+'.bin', dtype=dtype, mode='r',shape=shape)
    else:
        retrieved_data = np.memmap(folder+"/"+name+'.bin', dtype=dtype, mode='r')
    return retrieved_data

# def save_memory_mapped_indices(data, folder, name):
#     dt = np.dtype('i4,i4')  # Define the data type for tuples (two 32-bit integers)
#     shape = (len(data),)  # The shape is determined by the number of sublists
#     filepath = os.path.join(folder,name)+'.npy'
#     memmap_array = np.memmap(filepath, dtype=dt, shape=shape, mode='w+')
#     for i, sublist in enumerate(data):
#         memmap_array[i] = sublist
#     memmap_array.flush()

def save_memory_mapped_indices(idxs, folder, name):
    '''
    idxs: list of lists of indexes (pixels) : [ cams, poses, indexes (np.arr(n,2)) ]

    '''
    print("saving mmap")
    n_cams = len(idxs)
    # n_frames = len(idxs[0])
    lengths = [ [ j.shape[0] for j in i ] for i in idxs ]
    # print(idxs)
    lengths = [n_cams] + lengths
    list_flat = [ np.concatenate( i, axis = 0 ) for i in idxs ]
    idxs_flat = np.concatenate( list_flat, axis=0 )
    # idxs = np.concatenate( idxs, axis = 0 )
    idxs_flat = idxs_flat.astype(np.int32)
    idxs_flat = idxs_flat.flatten()
    save_memory_mapped_arr(idxs_flat, folder, name, flat=True)
    with open(folder+"/"+name+'.pickle', 'wb') as file:
        pickle.dump(lengths, file)

def load_memory_mapped_indices(folder, name):
    print("loading mmap")
    path_pickle = os.path.join(folder,name)+'.pickle'
    with open(path_pickle, 'rb') as file:
        lengths = pickle.load(file)
    length = lengths[0]
    lengths = lengths[1:]
    n_frames = len(lengths[0])

    data = load_memory_mapped_arr(folder,name,flat=True, dtype=np.int32)
    indices = []
    c = 0
    for j in range(length):
        indices.append([])
        for i,l in enumerate(lengths[j]):
            o = data[c:c+l*2]
            o = o.reshape([-1,2])
            indices[j].append(o)
            c+=l*2
    return indices
    





# You can now access the data as if it's in memory, but it's memory-mapped
if __name__ == "__main__":

    # # test arr
    # random_data = np.random.rand(*(10000,10000)).astype(np.int32)
    # save_memory_mapped_arr(random_data, ".", "ao")
    # r = load_memory_mapped_arr(".", "ao")
    # print(r.shape)
    # os.remove("./ao.bin")
    # os.remove("./ao.npy")

    # test indices
    indices = [[np.array([[1,2],[3,4],[5,6]],dtype=np.int32),np.array([[7,8],[9,10]],dtype=np.int32)],[np.array([[7,8]],dtype=np.int32),np.array([[7,8]],dtype=np.int32)]]
    print(indices)
    save_memory_mapped_indices(indices, ".", "idxs")
    indices_loaded = load_memory_mapped_indices(".","idxs")
    print(indices_loaded)
    os.remove("./idxs.pickle")
    os.remove("./idxs.bin")
