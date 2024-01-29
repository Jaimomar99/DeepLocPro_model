'''
Generate ESM-2 embeddings (per position) and save as one 
file per sequence. Use md5 hash of sequence as file name.
Adapted from DeepTMHMM and esm fsdp example.
'''
from hashlib import md5
from esm import FastaBatchedDataset, pretrained, FastaBatchedDataset
import torch
import os
import argparse
import pathlib



def hash_aa_string(string):
    return md5(string.encode()).digest().hex()

from tqdm.auto import tqdm
def generate_esm_embeddings(fasta_file, esm_embeddings_dir, repr_layers=48):
    #esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t48_15B_UR50D')


    dataset = FastaBatchedDataset.from_file(fasta_file)

    esm_model, esm_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    esm_model.eval()

    with torch.no_grad():
        if torch.cuda.is_available():
            #torch.cuda.set_device(1)
            esm_model = esm_model.cuda()

        batch_converter = esm_alphabet.get_batch_converter()
        
        print("Starting to generate embeddings")

        for idx, item in enumerate(tqdm(dataset)):
            
            label, seq = item
            
            if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt'):
                print("Already processed sequence")
                continue
                                
            #print(f"Sequence length: {len(original_aa_string)}")
            
            seqs = list([("seq", s) for s in [seq]])
            labels, strs, toks = batch_converter(seqs)

            out = None

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # minibatch_max_length = toks.size(1)
            out = esm_model(toks, repr_layers=[33], return_contacts=False)["representations"][33]
            out = out.cpu()
            
            # set nan to zeros
            out[out!=out] = 0.0

            res = out.transpose(0,1)[1:-1] 
            seq_embedding = res[:,0]

            output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt', 'wb')
            torch.save(seq_embedding, output_file)
            output_file.close()
            
            #print(f"Saved embedding to {esm_embeddings_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    args = parser.parse_args()


    generate_esm_embeddings(args.fasta_file, args.output_dir, repr_layers=33)

if __name__ == '__main__':
    main()