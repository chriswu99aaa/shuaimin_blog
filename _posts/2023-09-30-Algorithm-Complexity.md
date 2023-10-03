
## Week 1

### Directed Graph

Directed Graph has ordered pairs of vertices

### Union-Find

A *graph* is a pair G = (V,E), where V is a finite set and E a set of subsets of V of cardinality exatctly 2.

A *path* is a sequence of distinct vertices such that, for all i from 0 to less k.

```
underected st-CON

Given: A graph G=(V,E) and vertices s,t in V.
Return: Yes if t is reachable from s in G, No otherwise
```

```
Connectivity
Given: A graph G=(V,E)
Return: Yes if G is connected, No otherwise
```

A *Connected component* of a graph is a maximal set of vertices each of which is reachable from any other.

use depth-first search to find connected components of a graph G in linear time.

```
begin DFS(G, v, i) 
    mark v with i
    for each w ∈ G.edges(v) do 
        if w unmarked
            DFS(G, w, i)
```

```
begin CC(G) 
    i←0
    for v ∈ G.vertices 
        if v is unmarked 
            DFS(G, v, i)
            i←i+1
```

* makeSet(e): Create a singleton set containing the element e and name this set "e"
* union(A,B): Update A and B to create    AUB, naming the result as "A" or "B"
* find(e): Return the name of the set containg the element e.


#### List-Based Implementation for Disjoint Sets

```
makeSet():
    s = |v|
    s ->size = 1
    v -> cell = s
    add s to P
```

```
find(x):
    retunr x.head
```

```
union(s,t)
    if s->size > t->size
        remove t from P
        for v in t
            v ->cell = s
        s = append(s,t)
        s -> size = (s->size) + (t->size)
    else 
        do the same, but with s and t exchanged
```


```
union(u,v)
    if the set u is smaller than v then
        for each element x in the set u do
            remove x from u and addit v
            x.head <- v
    else
        for each element x in the set v do
            remove x from v and add it u
            x.head <- u
```
**Lemma:** In a series of operations of makesSet, union and find on n elements using the size-heuristic no element can have its cell field assigned more than round_down(log n)+1 times.

**Theorem:** With the above implementation the running time of union-find(G) is O(m + nlog(n))

**Theorem**: performing a sequence of m union and find operations, starting from n singleton sets, using the above list-based implementation of unionfind strucure, takes 
$$O(n log n + m) $$

```
union-find(V,E)
let P= ∅
for v ∈ V 
    makeSet(v )
for (u,v) ∈ E
    if find(u)  ̸= find(v)
        union(find(u),find(v ))
```

```
find(v)
    if v->parent is not a root
        v->parent = find(v->parent) //flattening step
    return v->parent
```

## Week 2

### Fast and Slow

A polynomial is an expression p(x) of the form

$$
a_nx^n + a_{n-1}x^{n-1}+...+a_1x+a_0
$$

A function f: N -> N is polynomially bounded f for some polynomial p $ f(n) <= p(n)$

A function f: N -> N is doubly exponentially bounded if for some polynomial p $f(n)<= 2^{2^p(n)} $

A function f: N -> N is k-tubly exponentially bounded exponentially bounded if for some polynomial p 

![image](../pictures/k-tubly.png)


#### Tree Implementation of Union-Find

**Theorem**

Using the tree implementation, the algorithm union-find(G=(V,E)) runs in $$O((n+m)\alpha(n))$$ time, where n=|V| and m=|E|

Union-by-size: Store with each node v the size of the subtree rooed at v, denoted by n(v). In a union, weow make the tree of the smaller set a subtree of the other tree, and update the size field of the root of the resulting tree.

Path Compression: In a find operaion, for each node v that the fid visits, reset the parent pointer from v to point to the root.

![image](../pictures/path_compre.png)

```
makeSet():
    for each singleton element x do
        x.parent <- x
        x.size <- 1
```

```

union(x,y):
    if x.size < y.size then
        x.parent <- y
        y.size <- y.size + x.size
    else
        y.parent <- x
        x.size <- y.size + x.size
```

```
find(x)
r <- x
while r.parent ≠ r do
    r <- r.pare

z <- x
while z.parent ≠ z do
    w <- z
    z <- z.parent
    w.parent <- r
```
r ← x: Initialize a variable r with the current element x.

while r.parent ≠ r do: Loop until r is its own parent, which means r is the root of the current set.

Inside the loop, r ← r.parent: Move r up the parent pointers until the root is found.
z ← x: Reset the variable z to the current element x for path compression.

while z.parent ≠ z do: Loop until z is its own parent, which means z is the root of the original set before path compression.

Inside the loop:
w ← z: Store the current element in a variable w.
z ← z.parent: Move z up the parent pointers until the root is found.
w.parent ← r: Set the parent pointer of w to the root r.
In summary, this algorithm finds the root of the set to which element x belongs (variable r) and performs path compression by updating the parent pointers of the elements on the path from x to the root r. Path compression optimizes future find operations by making the trees representing sets more balanced and shallow.


**Theorem:** Performing a sequence of m union and find operations, starting fom n singleton sets, using the above tree based implementation of a union find structure, takes **O(n + mlog(n))** time


