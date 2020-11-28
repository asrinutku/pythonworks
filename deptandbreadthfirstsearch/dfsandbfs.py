graph = {
  'A': ['B', 'C'],
  'B': ['D', 'E'],
  'C': ['E'],
  'D': [],
  'E': ['D']
}

graph2 = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}


def bfs(graph, baslangıcnode):
    # daha once gidilmis nodelar ıcın dızı
    dahaoncegidilenler = []
    # sırada gidilecek nodeların tutuldugu dızı
    sıra = [baslangıcnode]

    # gidilecek node kalmayana kadar işleme devam
    while sıra:

        node = sıra.pop(0)
        if node not in dahaoncegidilenler:
            dahaoncegidilenler.append(node)
            # bulundugumuz nodeun komsularını alıyoruz
            komsular = graph[node]

            # aldıgımız komsuları sıraya ekliyoruz
            for komsu in komsular:
                sıra.append(komsu)

    return dahaoncegidilenler



gidildilistesi = []

def dfs(graph,dugum):

    #seçilen node daha önce ziyaret edilmemişse gidildi listesine ekliyoruz
    if dugum not in gidildilistesi:
        gidildilistesi.append(dugum)

        for node in graph[dugum]:
            dfs(graph, node)



print("(graph 1 ile ) BFS : ",bfs(graph, 'A'))

dfs(graph2, 'A')
print("\n(graph 2 ile ) DFS : ",gidildilistesi)