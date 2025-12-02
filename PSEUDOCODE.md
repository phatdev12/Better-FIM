# Better-FIM Pseudocode

## Algorithm 1: The framework of Better-FIM

```python
Input: G = (V, E): social network; pop: the size of population; 
       g_max: the maximum number of iterations; cr: the probability of 
       crossover; mu: the probability of mutation; k: the size of the 
       seed set; λ: balance weight;

Output: A k-node set S with fair influence spread;

Step1: Community Detection.
1: C = {C₁, C₂, ..., Cₘ} ← obtain community structure by Leiden algorithm;

Step2: Population Initialization.
2: SN = {SN₁, SN₂, ..., SNₙ} ← calculate the scores of nodes 
   in G using PageRank;
3: SC = {SC₁, SC₂, ..., SCₘ} ← calculate the scores of communities in C;
4: P ← Initialization(G, pop, k, C, SN, SC);

Step3: Population Evolution.
5: Set iteration counter to zero: g = 0;
6: while g < g_max do
7:     Sort all individuals of P in descending order using F;
8:     P' ← Crossover(G, pop, k, C, SN, P, cr);
9:     P'' ← Mutation(G, pop, k, C, SN, P', mu);
10:    for i = 1 to pop do
11:        Pᵢ ← Max(Pᵢ, P''ᵢ, F);
12:    end for
13:    g = g + 1;
14: end while
15: S ← arg max_{Pᵢ ∈ P}(F(Pᵢ));
```

---

## Algorithm 2: Initialization(G, pop, k, C, SN, SC)

**Input:** G = (V, E): social network; pop: the size of population; k: the size of the seed set; C: community structure; SN: node scores; SC: community scores;

**Output:** An initialized population P;

1. Initialize P with empty vectors;
2. **for** i = 1 to pop **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;// Community-based solution
4. &nbsp;&nbsp;&nbsp;&nbsp;Initialize community counter CC = {0, 0, ..., 0}_m;
5. &nbsp;&nbsp;&nbsp;&nbsp;**for** j = 1 to k **do**
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Obtain the potential community C_t according to the probability calculated by SC scores;
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CC_t = CC_t + 1;
8. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
9. &nbsp;&nbsp;&nbsp;&nbsp;**for** h = 1 to m **do**
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** CC_h ≠ 0 **then**
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add the top CC_h nodes with the highest node scores in C_h to P_i;
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
13. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
14. &nbsp;&nbsp;&nbsp;&nbsp;Append P_i to P;
15. &nbsp;&nbsp;&nbsp;&nbsp;
16. &nbsp;&nbsp;&nbsp;&nbsp;// Fairness-aware random solution
17. &nbsp;&nbsp;&nbsp;&nbsp;Vectorize community selection: picks ← Random_Choice(C, size=k, p=Normalize(SC));
18. &nbsp;&nbsp;&nbsp;&nbsp;counts ← Count_Occurrences(picks);
19. &nbsp;&nbsp;&nbsp;&nbsp;**for** each (C_t, count) in counts **do**
20. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Select top count nodes by SN from C_t and add to P_i;
21. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
22. &nbsp;&nbsp;&nbsp;&nbsp;Fill remaining slots with top-SN nodes globally;
23. &nbsp;&nbsp;&nbsp;&nbsp;Append P_i to P;
24. &nbsp;&nbsp;&nbsp;&nbsp;
25. &nbsp;&nbsp;&nbsp;&nbsp;// SN-weighted random solution
26. &nbsp;&nbsp;&nbsp;&nbsp;P_i ← Random_Choice(V, size=k, p=Normalize(SN));
27. &nbsp;&nbsp;&nbsp;&nbsp;Append P_i to P;
28. **end for**

---

## Algorithm 3: Crossover(G, pop, k, C, SN, P, cr)

**Input:** G, pop, k, C, SN, P, cr;

**Output:** Population P' after crossover;

1. P' ← Copy(P);
2. **for** i = 1 to ⌊pop/2⌋ **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;**for** j = 1 to k **do**
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** random() < cr **then**
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Swap P'_i[j] ↔ P'_{pop-i}[j];
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
7. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
8. &nbsp;&nbsp;&nbsp;&nbsp;// Repair if needed: remove duplicates and fill with top-SN nodes
9. &nbsp;&nbsp;&nbsp;&nbsp;combined ← Set(P'_i) ∪ Set(P'_{pop-i});
10. &nbsp;&nbsp;&nbsp;&nbsp;sorted ← Sort_Descending(combined, by SN);
11. &nbsp;&nbsp;&nbsp;&nbsp;P'_i ← sorted[0:k];
12. **end for**
13. **return** P';

---

## Algorithm 3.1: Selection with Elitism + Random Injection

**Input:** P: current population; F: fitness function; pop: population size; k: seed set size; C: communities

**Output:** Population after selection and diversity injection

1. scores ← [(Pi, F(Pi)) for Pi in P]
2. sorted ← Sort_Descending(scores, by score)
3. elites ← [individual for (individual, _) in sorted[0:pop]]
4. r ← ⌊pop / 3⌋  // number of random individuals to inject
5. R ← []
6. **while** |R| < r **do**
7. &nbsp;&nbsp;&nbsp;&nbsp;S ← []
8. &nbsp;&nbsp;&nbsp;&nbsp;**while** |S| < k **do**
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c ← Random_Choice(C)  // random community
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v ← Random_Choice(c)  // random node in community c
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** v ∉ S **then** S.add(v)
12. &nbsp;&nbsp;&nbsp;&nbsp;**end while**
13. &nbsp;&nbsp;&nbsp;&nbsp;R.add(S)
14. **end while**
15. P_selected ← elites ∪ R  // Note: may exceed pop; next stage trims to pop
16. **return** P_selected

Notes:
- GPU path: compute scores on GPU and use argsort; random injection uses uniform selection over communities and their nodes.

---

## Algorithm 4: Mutation(G, pop, k, C, SN, P', mu, groups)

**Input:** G, pop, k, C, SN, P', mu, groups;

**Output:** Population P'' after mutation;

1. P'' ← Copy(P');
2. **for** i = 1 to pop **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;**if** random() < mu **then**
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Remove random node
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;idx ← Random_Int(0, k-1);
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;removed_node ← P''_i[idx];
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Remove P''_i[idx];
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Calculate current group coverage
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** each group g in groups **do**
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;coverage[g] ← Count_Nodes_In_Group(P''_i, g);
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end for**
13. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
14. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Compute fairness-weighted community scores
15. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** each community c in C **do**
16. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;weight[c] ← 0;
17. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** each group g in groups **do**
18. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;overlap ← |c ∩ g|;
19. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ideal_count ← max(1, ⌊k × |g| / |V|⌋);
20. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;deficit ← max(0, ideal_count - coverage[g]);
21. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;weight[c] ← weight[c] + deficit × overlap;
22. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end for**
23. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;weight[c] ← weight[c] + 1;  // Avoid zero weight
24. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end for**
25. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
26. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Select community by fairness-weighted probability
27. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;probs ← Normalize(weight);
28. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C_selected ← Random_Choice(C, p=probs);
29. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
30. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Select node from community by SN-weighted probability
31. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;candidates ← {v ∈ C_selected | v ∉ P''_i};
32. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** candidates ≠ ∅ **then**
33. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SN_probs ← Normalize([SN[v] for v in candidates]);
34. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;new_node ← Random_Choice(candidates, p=SN_probs);
35. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add new_node to P''_i;
36. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**else**
37. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add removed_node to P''_i;  // Fallback
38. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
39. &nbsp;&nbsp;&nbsp;&nbsp;**end if**
40. **end for**
41. **return** P'';

---

## Algorithm 5: Fitness Evaluation (F)

**Input:** S: seed set; G: graph; groups: attribute groups; ideal_influences: ideal influence per group; λ: balance weight; p: propagation probability; mc: Monte Carlo simulations;

**Output:** Fitness score F(S);

1. **for** each group g_i in groups **do**
2. &nbsp;&nbsp;&nbsp;&nbsp;actual_influence[g_i] ← Run_IC(G, S, p, mc, target=nodes_in_g_i);
3. &nbsp;&nbsp;&nbsp;&nbsp;fraction[g_i] ← actual_influence[g_i] / |g_i|;
4. &nbsp;&nbsp;&nbsp;&nbsp;violation[g_i] ← max(0, (ideal_influences[g_i] - actual_influence[g_i]) / ideal_influences[g_i]);
5. **end for**
6. MF ← min(fraction);
7. DCV ← mean(violation);
8. F(S) ← λ × MF - (1-λ) × DCV;
9. **return** F(S);

---

## Algorithm 6: Run_IC with Live-Edge Caching

**Input:** G, S, p, mc, target_nodes;

**Output:** Average influence spread;

1. live_graphs ← Get_Live_Edge_Samples(G, p, mc);  // Cached globally
2. total_spread ← 0;
3. **for** each live_graph in live_graphs **do**
4. &nbsp;&nbsp;&nbsp;&nbsp;active ← Set(S);
5. &nbsp;&nbsp;&nbsp;&nbsp;queue ← List(S);
6. &nbsp;&nbsp;&nbsp;&nbsp;**while** queue ≠ ∅ **do**
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;u ← queue.pop_front();
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** each v in live_graph.neighbors(u) **do**
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** v ∉ active **then**
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;active.add(v);
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;queue.append(v);
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
13. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end for**
14. &nbsp;&nbsp;&nbsp;&nbsp;**end while**
15. &nbsp;&nbsp;&nbsp;&nbsp;**if** target_nodes specified **then**
16. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;spread ← |active ∩ target_nodes|;
17. &nbsp;&nbsp;&nbsp;&nbsp;**else**
18. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;spread ← |active|;
19. &nbsp;&nbsp;&nbsp;&nbsp;**end if**
20. &nbsp;&nbsp;&nbsp;&nbsp;total_spread ← total_spread + spread;
21. **end for**
22. **return** total_spread / mc;



---

## Độ phức tạp

- **Khởi tạo quần thể**: O(POP_SIZE × k × |V|)
- **Mỗi thế hệ GA**:
  - Đánh giá: O(POP_SIZE × |groups| × mc × |V|) với caching → giảm đáng kể
  - Lai ghép/đột biến: O(POP_SIZE × k)
- **Tổng**: O(MAX_GEN × POP_SIZE × |groups| × mc × |V|)

## Tối ưu hóa chính

1. **Live-Edge Caching**: Pre-sample mc đồ thị live-edge, tái sử dụng cho tất cả tính toán IC
2. **Influence Memoization**: Cache kết quả IC theo (seed_set, group_id) trong mỗi thế hệ
3. **Vector hóa**: Dùng NumPy cho chọn cộng đồng, sort nodes, giảm vòng lặp Python
4. **Greedy Ideal với Live-Edge**: Dùng cùng live-edge samples cho greedy selection

## Tham số mặc định

- k = 40
- POP_SIZE = 10
- MAX_GEN = 150
- P_CROSSOVER = 0.6
- P_MUTATION = 0.1
- λ = 0.5
- p = 0.01
- mc = 1000 (evaluation), 30 (ideal influence)
