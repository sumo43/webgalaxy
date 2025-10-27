# WebGalaxy

```python

python3 summarize_first_10k.py   --dataset Plugiloinc/45_Million_Websites   --split train   --limit 10000   --out website_summaries_first10k.jsonl   --concurrency 16
python3 cluster_websites.py --input website_summaries_first10k.jsonl --output_dir out_clusters --k 40 --model_name nvidia/NV-Embed-v2
cp out_clusters/* . 
python3 -m http.server 8080 # run http server

```
