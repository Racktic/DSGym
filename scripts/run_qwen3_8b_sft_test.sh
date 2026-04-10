cd /data/fnie/qixin/DSGym                                                                                                                            
dsgym eval \                                                                                                                                         
    --model /data/fnie/LLaMA-Factory/saves/qwen3-8b-aide-v5/full/sft/checkpoint-50 \                                                                 
    --dataset dspredict-easy \                                                                                                                       
    --backend vllm \
    --agent aide \                                                                                                                                   
    --num-drafts 5 \                                                                                                                                 
    --max-turns 20 \
    --max-workers 2 \                                                                                                                                
    --limit 2 \                                                                                                                                      
    --best-node-strategy latest \
    --output-dir /data/fnie/qixin/DSGym/evaluation_results/qwen3_8b_sft_easy_test 