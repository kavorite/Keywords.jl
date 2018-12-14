module Keywords

using WordTokenizers
using Embeddings
using LightGraphs
using SimpleWeightedGraphs
using LinearAlgebra

export keywords, dictionary, cos

function dictionary(embeddings::Embeddings.EmbeddingTable)
    Dict((embeddings.vocab[i], embeddings.embeddings[:, i])
         for i in 1:length(embeddings.vocab))
end

function Base.cos(a::Vector, b::Vector) a ⋅ b / norm(a) / norm(b) end

function termAdjacency(text::AbstractString, D, S=Set())
    T = [t for t ∈ unique!(tokenize(text))
         if haskey(D, t) && t ∉ S]
    n = length(T)
    M = [cos(D[t], D[w]) for t ∈ T for w ∈ T]
    M = reshape(M, (n, n))
    M = SparseMatrixCSC(M)
    for i ∈ 1:n  M[i, :] ./= sum(M[i, :]) end
    (M, T, n)
end

function keywords(text, embeddings; α=0.85, k=Inf, ϵ=1e-6, stops=Set())
    T = [t for t ∈ unique!(tokenize(text)) if haskey(embeddings, t) && t ∉ stops]
    M = SimpleWeightedDiGraph(length(T))
    for (i, t) ∈ enumerate(T)
        for (j, w) ∈ enumerate(T[[1:i-1; i+1:length(T)]])
            add_edge!(M, i, j, cos(embeddings[t], embeddings[w]))
        end
    end
    r = pagerank(M)
    sorted = sort(1:length(T); by=(i -> r[i]), rev=true)
    T[sorted], r[sorted]
end

end
