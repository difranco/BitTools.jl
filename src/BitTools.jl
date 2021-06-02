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

end # module
