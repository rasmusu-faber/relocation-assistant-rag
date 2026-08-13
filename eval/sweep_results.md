| Embedding model | Chunk size | hit@1 | hit@4 | MRR@4 | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | 400 | 88% | 100% | 0.938 | 60 |
| `all-MiniLM-L6-v2` | 800 | 94% | 100% | 0.969 | 35 |
| `all-MiniLM-L6-v2` | 1200 | 100% | 100% | 1.000 | 22 |
| `bge-small-en-v1.5` | 400 | 100% | 100% | 1.000 | 60 |
| `bge-small-en-v1.5` | 800 | 100% | 100% | 1.000 | 35 |
| `bge-small-en-v1.5` | 1200 | 100% | 100% | 1.000 | 22 |

_16 eval questions; overlap = 15% of chunk size; embeddings L2-normalized, cosine space; models compared out-of-the-box (no query-side instructions)._
