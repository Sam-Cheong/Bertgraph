import argparse
import numpy as np

from tcr_predictor_fixed import TCRPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./results/my_tcr_bert.pt",
                        help="Path to local TCR-BERT .pt checkpoint")
    parser.add_argument("--cdr3", type=str, default="CASSSQGLAGQPQHF",
                        help="A CDR3 sequence to embed")
    parser.add_argument("--expect_dim", type=int, default=256,
                        help="Expected embedding dimension (d_model)")
    args = parser.parse_args()

    predictor = TCRPredictor(model_name=args.model)
    # 注意方法名是 embed_sequences
    emb = predictor.embed_sequences([args.cdr3])  # shape should be (1, d_model)
    if hasattr(emb, "shape"):
        shape = emb.shape
    else:
        # 兼容返回 list/torch.Tensor 等
        try:
            import torch
            if isinstance(emb, torch.Tensor):
                shape = tuple(emb.shape)
                emb = emb.detach().cpu().numpy()
            else:
                emb = np.asarray(emb)
                shape = emb.shape
        except Exception:
            emb = np.asarray(emb)
            shape = emb.shape

    print(f"Embedding shape: {shape}")
    if len(shape) == 2 and shape[0] == 1:
        if shape[1] == args.expect_dim:
            print(f"OK: got (1, {args.expect_dim}) — local TCR-BERT replacement works.")
        else:
            print(f"WARNING: got (1, {shape[1]}), expected (1, {args.expect_dim}). "
                  "Please ensure your ImmuneBERT d_model matches downstream.")
    else:
        print("WARNING: unexpected embedding shape; check your embedder return type and batching.")

if __name__ == "__main__":
    main()