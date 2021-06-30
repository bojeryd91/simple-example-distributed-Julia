using Distributed, TickTock, Printf

const IN_SLURM = "SLURM_JOBID" in keys(ENV)

# load packages
IN_SLURM && using ClusterManagers

# Here we create our parallel julia processes
if IN_SLURM
    pids = addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
    print("\nIn SLURM\n")
    println(pids)
else
    pids = addprocs()
    println("Not SLURM")
end

@everywhere using Optim, Base.Threads
@everywhere function a_task(a, b)
    f(x) = a*x[1] + b*x[2] + (a+b)*x[1]^2 + 2.0*x[2]^2
    res = optimize(f, [a, b])
    return(res.minimum)
end

function no_PL(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = ones(A_L, B_L) #SharedArray{Float64}(A_L, B_L)

    @inbounds for (i_a, i_b) in [(i_a, i_b)
                    for i_a in eachindex(A), i_b in eachindex(B)]
        R[i_a, i_b] = a_task(A[i_a], B[i_b])
    end
    return(R)
end
function w_threads(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = ones(A_L, B_L) #SharedArray{Float64}(A_L, B_L)

    @threads for (i_a, i_b) in [(i_a, i_b)
                    for i_a in eachindex(A), i_b in eachindex(B)]
        R[i_a, i_b] = a_task(A[i_a], B[i_b])
    end
    return(R)
end

function w_distributed(A_L, B_L)
    A = range(0.0, 20.0, length=A_L)
    B = range(0.0, 50.0, length=B_L)
    R = ones(A_L, B_L) #SharedArray{Float64}(A_L, B_L)

    @sync @distributed for (i_a, i_b) in [(i_a, i_b)
                    for i_a in eachindex(A), i_b in eachindex(B)]
        R[i_a, i_b] = a_task(A[i_a], B[i_b])
    end
    return(R)
end

no_PL(2, 2)
w_threads(2, 2)
w_distributed(2, 2)

funcs = [no_PL, w_threads, w_distributed]
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

rmprocs(pids)
