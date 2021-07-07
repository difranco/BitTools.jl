module BitTools

export rbitvec, invert_at_indices

using RandomNumbers
using Random
using Distributions

rng = Xorshifts.Xoshiro128Plus(0xdeadbeef)

@inline function rbitvec(len, occ)
    # Returns a random bit vector of length len
    # with occ one bits and len-occ zero bits
    return BitVector(shuffle!(rng, [zeros(Bool, len - occ); ones(Bool, occ)]))
end

using Distributions: Beta
beta = Beta(5, 5)

@inline function rbitvec(len)
    # Returns a random bit vector of length len
    # and occupancy Beta(5,5) between 1 and len
    return rbitvec(len, Int(floor(len * rand(rng, beta)) + 1))
end

@inline function invert_at_indices(x::BitVector, inds)
    # returns copy of x with bits flipped at positions given in inds
    out = copy(x)
    map!(!, view(out, inds), x[inds])
    return out
end

using Clustering
import StatsBase: sample, Weights
using Distributions

export findClustering, sample, sampleWeights

function hammingDistances(data :: BitMatrix)
    (dim, len) = size(data)
    r = zeros(UInt8, len, len)
    for i in 1:len
        for j in 1:i
            r[i,j] = r[j,i] = sum(data[:,i] .!= data[:,j])
        end
    end
    return r
end

@inline score(r, dists) = sum(silhouettes(r, dists))

function tryNewClustering(k :: Int, oldClustering :: KmeansResult, data :: BitMatrix)
    (dim, oldK) = size(oldClustering.centers)
    if k == oldK
        return oldClustering
    end
    c = copy(oldClustering.centers)

    if k < oldK
        c = c[:, 1:k]
    elseif k > oldK
        c = hcat(c, rand(dim, k - oldK))
    end

    return kmeans!(data, c, maxiter = 30)
end

function findClustering(data :: BitMatrix)
    (dim, len) = size(data)
    initialK = Int(max(floor(sqrt(sqrt(len))), 2))
    initialGuess = kmeans(data, initialK, maxiter = 50)
    dists = hammingDistances(data)

    bestGuess = initialGuess
    bestGuessScore = score(initialGuess, dists)

    for k in initialK + 1 : Int(floor(1.5*sqrt(len)))
        newGuess = tryNewClustering(k, bestGuess, data)
        newGuessScore = score(newGuess, dists)

        if(newGuessScore > bestGuessScore)
            bestGuess = newGuess
            bestGuessScore = newGuessScore
        end
    end

    return bestGuess
end

function sampleWeights(c :: KmeansResult, numInSamples = 10)
    idxs = collect(1:size(c.centers)[2])
    points = sum(c.counts)
    cols = sample(idxs, Weights(c.counts ./ points), numInSamples)
    len = length(cols)
    return sum(c.centers[:, cols], dims = 2) ./ len
end

function sample(c :: KmeansResult, numInSamples = 10, numOutSamples = 1)
    weights = sampleWeights(c, numInSamples)
    out = BitMatrix(undef, length(weights), numOutSamples)
    for i in 1: length(weights)
        for j in 1:numOutSamples
            out[i, j] = rand(Bernoulli(weights[i]))
        end
    end
    return out
end

end # module
