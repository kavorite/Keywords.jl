module Keywords

using WordTokenizers
using LinearAlgebra
using SparseArrays
using Embeddings

export keywordize, dictionary, cos

function dictionary(embeddings::Embeddings.EmbeddingTable)
    Dict((embeddings.vocab[i], embeddings.embeddings[:, i])
         for i in 1:length(embeddings.vocab))
end

function Base.cos(a::Vector, b::Vector) a ⋅ b / norm(a) / norm(b) end

function mkM(text::AbstractString, D, S=Set())
    T = unique!(tokenize(text))
    T = [t for t ∈ T if haskey(D, t) && t ∉ S]
    n = length(T)
    M =  reshape([cos(D[t], D[w])
                  for t ∈ T for w ∈ T], (n, n))
    (M, T, n)
end

# PageRank
function keywordize(text, embeddings; α=0.85, k=Inf, ϵ=1e-8, S=Set())
    # create a symmetric matrix of cosine similarities between terms
    M, T, n = mkM(text, embeddings, S)
    n = length(T)
    r′ = fill(1/n, (1, n))
    r = ones(1, n)
    M′ = (α * M) + (((1 - α) / n) * ones(n, n))
    i = 1
    # power method: asymptotic convergence
    while norm(r′ - r, 2) > ϵ
        i <= k || error(StackOverflowError())
        r = r′
        r′ *= M′
        i += 1
    end
    T[sort(1:n; by=(i -> r′[i]), rev=true)]
end

end
