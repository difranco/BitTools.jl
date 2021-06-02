using BitTools, Test

@testset "rbitvec" begin
    l = 5
    @test length(rbitvec(l)) == l
end

@testset "invert_at_indices" begin
    b = BitVector([1, 0, 1, 0])
    @test invert_at_indices(b, [3,4]) == BitVector([1, 0, 0, 1])
end
