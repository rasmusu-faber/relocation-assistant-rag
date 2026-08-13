| Eval set | Variant | hit@4 | hit@1 | MRR@4 |
| --- | --- | ---: | ---: | ---: |
| easy (distinct) (16 q) | baseline | 100% | 94% | 0.969 |
| easy (distinct) (16 q) | + rerank | 100% | 94% | 0.969 |
| hard (siblings) (18 q) | baseline | 100% | 72% | 0.852 |
| hard (siblings) (18 q) | + rerank | 100% | 89% | 0.935 |

_Candidate pool = 20, top-k = 4, cross-encoder = `cross-encoder/ms-marco-MiniLM-L-6-v2`. Re-ranking only re-orders the retriever's pool, so hit@4 is unchanged; hit@1 / MRR show whether the right document is pulled to the top._
