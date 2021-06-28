using Base.Threads, Distributed, BenchmarkTools, TickTock,
        Printf

const IN_SLURM = "SLURM_JOBID" in keys(ENV)

# load packages
using Distributed
IN_SLURM && using ClusterManagers

# Here we create our parallel julia processes
if IN_SLURM
    pids = addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
    print("\n")
else
    pids = addprocs()
end

@everywhere using Optim, SharedArrays
@everywhere function a_task(a, b)
    f(x) = a*x[1] + b*x[2] + (a+b)*x[1]^2 + 2.0*x[2]^2
    res = optimize(f, [a, b])
    return(res.minimum)
end

function no_PL(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = ones(length(A), length(B))

    @inbounds for (i_a, i_b) in [(i_a, i_b)
                    for i_a in eachindex(A), i_b in eachindex(B)]
        R[i_a, i_b] = a_task(A[i_a], B[i_b])
    end
    return(R)
end
function w_threads(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = SharedArray{Float64}(A_L, B_L)

    @threads for (i_a, i_b) in [(i_a, i_b)
                    for i_a in eachindex(A), i_b in eachindex(B)]
        R[i_a, i_b] = a_task(A[i_a], B[i_b])
    end
    return(R)
end

function w_distributed(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = SharedArray{Float64}(A_L, B_L)
    @sync @distributed for (i_a, i_b) in [(i_a, i_b)
                    for i_a in eachindex(A), i_b in eachindex(B)]
        R[i_a, i_b] = a_task(A[i_a], B[i_b])
    end
    return(R)
end

function w_combined(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = SharedArray{Float64}(A_L, B_L)
    @sync @distributed for i_a in eachindex(A)
        @threads for i_b in eachindex(B)
            R[i_a, i_b] = a_task(A[i_a], B[i_b])
        end
    end
    return(R)
end
no_PL(2, 2)
w_threads(2, 2)
w_distributed(2, 2)
w_combined(2, 2)

funcs = [no_PL, w_threads, w_distributed, w_combined]
ticks = zeros(length(funcs))
for (t, func) in zip(eachindex(ticks), funcs)
    tick()
    for i = 1:5
        func(3000, 40)
    end
    ticks[t] = tok()
end

println("Run 3000×40:")
@printf("Linear:      %3.3g s\n", ticks[1]/5)
@printf("Threads:     %3.3g s\n", ticks[2]/5)
@printf("Distributed: %3.3g s\n", ticks[3]/5)
@printf("Combined:    %3.3g s\n", ticks[4]/5)

for (t, func) in zip(eachindex(ticks), funcs)
    tick()
    for i = 1:5
        func(30, 4000)
    end
    ticks[t] = tok()
end

println("Run 30×4000:")
@printf("Linear:      %3.3g s\n", ticks[1]/5)
@printf("Threads:     %3.3g s\n", ticks[2]/5)
@printf("Distributed: %3.3g s\n", ticks[3]/5)
@printf("Combined:    %3.3g s\n", ticks[4]/5)

for (t, func) in zip(eachindex(ticks), funcs)
    tick()
    for i = 1:5
        func(300, 400)
    end
    ticks[t] = tok()
end

println("Run 300×400:")
@printf("Linear:      %3.3g s\n", ticks[1]/5)
@printf("Threads:     %3.3g s\n", ticks[2]/5)
@printf("Distributed: %3.3g s\n", ticks[3]/5)
@printf("Combined:    %3.3g s\n", ticks[4]/5)

rmprocs(pids)
