module Keywords

import WordTokenizers.tokenize

export keywordize, cos

function Base.cos(a::Vector, b::Vector) a ⋅ b / norm(a) / norm(b) end

function keywordize(text, embeddings; α=0.85, k=Inf, ϵ=1e-8, S=Set())
    # create a symmetric matrix of cosine similarities between terms
    D = embeddings
    T = [t for t ∈ unique(tokenize(text)) if haskey(D, t) && t ∉ S]
    n = length(T)
    M = Symmetric(SparseMatrixCSC(
        reshape([cos(D[t], D[w])
                for t ∈ T for w ∈ T], (n, n))))
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
