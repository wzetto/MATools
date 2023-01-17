import Pkg; Pkg.add("PyCall"); Pkg.add("JLD"); Pkg.add("LinearAlgebra");
Pkg.add("Random"); Pkg.add("BenchmarkTools")

using PyCall
using JLD
using LinearAlgebra
using Random
using Base
using BenchmarkTools

"""
Define all the hyperparameters in this file.
"""
use_pbc = true #* Use pbc condition or not

if use_pbc
    pbc_term = "_pbc"
else
    pbc_term = ""
end

#* Suit with the Julia's format
pth = "/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/MATools/msad/mc/ver2/utils/ind_pair$pbc_term.jld"
ind_1nn = load(pth, "ind_1nn$pbc_term") .+1
ind_2nn = load(pth, "ind_2nn$pbc_term") .+1
ind_3nn = load(pth, "ind_3nn$pbc_term") .+1
ind_4nn = load(pth, "ind_4nn$pbc_term") .+1
ind_5nn = load(pth, "ind_5nn$pbc_term") .+1
ind_6nn = load(pth, "ind_6nn$pbc_term") .+1

ind_book = [
    ind_1nn, ind_2nn, ind_3nn, ind_4nn, ind_5nn, ind_6nn
]

cr_, mn_, co_, ni_ = 0.4, 0.05, 0.3, 0.25
target_val = 100
temperature = 3

benchmark_test = false #* Display the execution time during iteration.
debug_test = false #* Display the correlation function during each iteration.
cutoff_iter = 200_000

atom_get() = cr_, mn_, co_, ni_ 

function abs_dis(a, b, target)
    return abs(norm(a-b)-target)
end

phi1(x) = 2/sqrt(10)*x

phi2(x) = -5/3 + 2/3*(x^2)

phi3(x) = -17/30*sqrt(10)*x + sqrt(10)/6*(x^3)

function cpr(val1, val2)
    p1v1 = phi1(val1)
    p2v1 = phi2(val1)
    p3v1 = phi3(val1)
    p1v2 = phi1(val2)
    p2v2 = phi2(val2)
    p3v2 = phi3(val2)
    c11 = p1v1*p1v2
    c12 = (p1v1*p2v2+p2v1*p1v2)/2
    c13 = (p1v1*p3v2+p3v1*p1v2)/2
    c22 = p2v1*p2v2
    c23 = (p2v1*p3v2+p3v1*p2v2)/2
    c33 = p3v1*p3v2

    return [c11, c12, c13, c22, c23, c33]
end

#* Broadcast across cpr_book and num_element
cpr_book = hcat(cpr(2,2),cpr(1,1),cpr(-1,-1),cpr(-2,-2),cpr(2,-1),cpr(2,1),cpr(-1,1),cpr(2,-2),cpr(1,-2),cpr(-1,-2))'
function ideal_cor(cr_, mn_, co_, Nnn, n_a=192*27, mode="printNG")
    ni_ = 1-cr_-mn_-co_
    bond_num = size(Nnn)[1]
    #* Ideal number of atoms
    cr_i, mn_i, co_i, ni_i = n_a*cr_, n_a*mn_, n_a*co_, n_a*ni_

    #* Strict condition.
    # num_crcr = cr_i*(cr_i-1)/n_a/(n_a-1)*bond_num
    # num_mnmn = mn_i*(mn_i-1)/n_a/(n_a-1)*bond_num
    # num_coco = co_i*(co_i-1)/n_a/(n_a-1)*bond_num
    # num_nini = ni_i*(ni_i-1)/n_a/(n_a-1)*bond_num
    # num_crco = 2*cr_i*co_i/n_a/(n_a-1)*bond_num
    # num_crmn = 2*cr_i*mn_i/n_a/(n_a-1)*bond_num
    # num_coni = 2*co_i*ni_i/n_a/(n_a-1)*bond_num
    # num_comn = 2*co_i*mn_i/n_a/(n_a-1)*bond_num
    # num_crni = 2*cr_i*ni_i/n_a/(n_a-1)*bond_num
    # num_mnni = 2*mn_i*ni_i/n_a/(n_a-1)*bond_num

    #* Tolerant condition.
    num_crcr = cr_*cr_*bond_num
    num_mnmn = mn_*mn_*bond_num
    num_coco = co_*co_*bond_num
    num_nini = ni_*ni_*bond_num
    num_crco = 2*cr_*co_*bond_num
    num_crmn = 2*cr_*mn_*bond_num
    num_coni = 2*co_*ni_*bond_num
    num_comn = 2*co_*mn_*bond_num
    num_crni = 2*cr_*ni_*bond_num
    num_mnni = 2*mn_*ni_*bond_num

    #*2, 1, -1, -2: Cr, Mn, Co, Ni; Broadcast operation -> 10x6 matrix
    num_list = [
        num_crcr, num_mnmn, num_coco, num_nini,
        num_crco, num_crmn, num_comn, num_crni, num_mnni, num_coni
    ]
    cor_func = sum(num_list.*cpr_book, dims=1)
    # cor_func = (num_crcr*cpr(2,2)
    #            +num_mnmn*cpr(1,1)
    #            +num_coco*cpr(-1,-1)
    #            +num_nini*cpr(-2,-2)
    #            +num_crco*cpr(2,-1)
    #            +num_crmn*cpr(2,1)
    #            +num_comn*cpr(-1,1)
    #            +num_crni*cpr(2,-2)
    #            +num_mnni*cpr(1,-2)
    #            +num_coni*cpr(-1,-2))
               
    if cmp("printPLZ", mode) == 0
        println("ideal cor func of Cr",cr_*100,"Co",co_*100,"Ni",ni_*100,": ",cor_func)
    end

    return reshape(cor_func, (6,1))
end

ideal_mat = hcat(
    ideal_cor(cr_, mn_, co_, ind_1nn), 
    ideal_cor(cr_, mn_, co_, ind_2nn),
    ideal_cor(cr_, mn_, co_, ind_3nn),
    ideal_cor(cr_, mn_, co_, ind_4nn),
    ideal_cor(cr_, mn_, co_, ind_5nn),
    ideal_cor(cr_, mn_, co_, ind_6nn))'

# if benchmark_test
#     println(ideal_mat)
# end

function cor_func(ind_nNN, ele_list)
    ele_list = repeat(ele_list, 27) #* in PBC condition.
    cor_func_n = zeros(Float64, 6, 1)
    for i in 1:size(ind_nNN)[1]
        i1, i2 = ind_nNN[i,:]
        a1, a2 = ele_list[i1], ele_list[i2]
        cor_f = cpr(a1, a2)
        cor_func_n += cor_f
    end

    return cor_func_n
end

function ele_list_gen(cr_c, mn_c, co_c, ni_c, mode="randchoice")
    @assert abs(cr_c+mn_c+co_c+ni_c-1) < 0.001 "Make sure the sum of atomic contents = 1"

    while true
        if cmp("randchoice", mode) == 0
            len_cr = rand(range(convert(Int, round(cr_c*192)), convert(Int, round(cr_c*192))+1, step=1))
            len_mn = rand(range(convert(Int, round(mn_c*192)), convert(Int, round(mn_c*192))+1, step=1))
            len_co = rand(range(convert(Int, round(co_c*192)), convert(Int, round(co_c*192))+1, step=1))
        elseif cmp("int", mode) == 0
            len_cr = convert(Int, round(cr_c*192))
            len_mn = convert(Int, round(mn_c*192))
            len_co = convert(Int, round(co_c*192))
        end

        len_ni = 192-len_cr-len_mn-len_co
        if abs(len_ni-192*ni_c) <= 1
            cr_list = zeros(Float64, len_cr) .+ 2
            mn_list = zeros(Float64, len_mn) .+ 1
            co_list = zeros(Float64, len_co) .- 1
            ni_list = zeros(Float64, len_ni) .- 2
            ele_list_raw = cat(cr_list, mn_list, co_list, ni_list, dims=1)
            return reshape(shuffle(ele_list_raw), (192,1))
            break
        end
    end
end

function cor_func_all(state, test=false)
    cor1 = cor_func(ind_1nn, state)
    cor2 = cor_func(ind_2nn, state)
    cor3 = cor_func(ind_3nn, state)
    cor4 = cor_func(ind_4nn, state)
    cor5 = cor_func(ind_5nn, state)
    cor6 = cor_func(ind_6nn, state)

    cor_ = hcat(cor1, cor2, cor3, cor4, cor5, cor6)'
    #* 6x6
    if test
        return norm(cor_-ideal_mat)
    else
        return cor_
    end
end

"""
Return the row indices
[[i1, j1], ..., [i1, jm]] [[i2, j1'], ..., [i2, jn']] -> raw 
[[i2, j1], ..., [i2, jm]] [[i1, j1'], ..., [i1, jn']] -> new

Then update cor_func_res with cpr(new) - cpr(raw)
return the residue - which is the "minus reward"
"""
function cor_func_embed(state, action)
    i1, i2 = action
    a1, a2 = state[i1], state[i2]

    if i1 == 192
        i1 = 0
    end

    if i2 == 192
        i2 = 0
    end

    state = repeat(state, 27) #* in PBC condition.

    #* Row indices, say: thank you, chatGPT
    ind_1nn_raw1 = [i[1] for i in findall(any(ind_1nn .% 192 .== i1, dims=2))]
    # ind_1nn_new1 = replace(ind_1nn_raw1, i1=>i2)
    ind_1nn_raw2 = [i[1] for i in findall(any(ind_1nn .% 192 .== i2, dims=2))]
    # ind_1nn_new2 = replace(ind_1nn_raw2, i2=>i1)
    # ind_1nn_raw = vcat(ind_1nn_raw1, ind_1nn_raw2)
    # ind_1nn_new = vcat(ind_1nn_new1, ind_1nn_new2)

    ind_2nn_raw1 = [i[1] for i in findall(any(ind_2nn .% 192 .== i1, dims=2))]
    ind_2nn_raw2 = [i[1] for i in findall(any(ind_2nn .% 192 .== i2, dims=2))]

    ind_3nn_raw1 = [i[1] for i in findall(any(ind_3nn .% 192 .== i1, dims=2))]
    ind_3nn_raw2 = [i[1] for i in findall(any(ind_3nn .% 192 .== i2, dims=2))]

    ind_4nn_raw1 = [i[1] for i in findall(any(ind_4nn .% 192 .== i1, dims=2))]
    ind_4nn_raw2 = [i[1] for i in findall(any(ind_4nn .% 192 .== i2, dims=2))]  

    ind_5nn_raw1 = [i[1] for i in findall(any(ind_5nn .% 192 .== i1, dims=2))]
    ind_5nn_raw2 = [i[1] for i in findall(any(ind_5nn .% 192 .== i2, dims=2))]
    
    ind_6nn_raw1 = [i[1] for i in findall(any(ind_6nn .% 192 .== i1, dims=2))]
    ind_6nn_raw2 = [i[1] for i in findall(any(ind_6nn .% 192 .== i2, dims=2))]

    raw_ind1 = [ind_1nn_raw1, ind_2nn_raw1, ind_3nn_raw1, ind_4nn_raw1, ind_5nn_raw1, ind_6nn_raw1]
    raw_ind2 = [ind_1nn_raw2, ind_2nn_raw2, ind_3nn_raw2, ind_4nn_raw2, ind_5nn_raw2, ind_6nn_raw2]

    cor_func_res = zeros(Float64, (6, 6))
    new_pair1, new_pair2 = zeros(Float64, 2), zeros(Float64, 2)
    raw_pair1, raw_pair2 = zeros(Float64, 2), zeros(Float64, 2)

    for ind in 1:6
        raw_ind_chosen1 = raw_ind1[ind]
        raw_ind_chosen2 = raw_ind2[ind]
        ind_nnchosen = ind_book[ind]

        for i in 1:size(raw_ind_chosen1)[1]
            #* For a1
            ind_i = raw_ind_chosen1[i]
            i1_raw, i2_raw = ind_nnchosen[ind_i, :]
            a1_raw, a2_raw = state[i1_raw], state[i2_raw]
            raw_pair1 = [a1_raw, a2_raw]
            new_pair1 = copy(raw_pair1)

            ind_a1 = findfirst(raw_pair1 .== a1)
            
            try
                new_pair1[ind_a1] = a2
            catch
                println(raw_pair1, a1, size(raw_ind_chosen1))
            end
            #* Update the res function.
            cor_func_res[ind,:] += (cpr(new_pair1[1],new_pair1[2])-cpr(raw_pair1[1],raw_pair1[2]))

        end

        for i in 1:size(raw_ind_chosen2)[1]
            #* For a2 
            ind_i = raw_ind_chosen2[i]
            i1_raw, i2_raw = ind_nnchosen[ind_i, :]
            a1_raw, a2_raw = state[i1_raw], state[i2_raw]
            raw_pair2 = [a1_raw, a2_raw]
            new_pair2 = copy(raw_pair2)

            ind_a2 = findfirst(raw_pair2 .== a2)
            new_pair2[ind_a2] = a1

            cor_func_res[ind,:] += (cpr(new_pair2[1],new_pair2[2])-cpr(raw_pair2[1],raw_pair2[2]))
        end
        
        # println(size(cor_func_res[ind,:]), size(cor_f))
    end

    return cor_func_res
end
    
function swap_step(action, cor_func_n, state, target_val)

    cor_func_raw = cor_func_n
    action_ = copy(action)
    state_ = copy(state)

    cor_func_res = cor_func_embed(state, action) #* 6x6
    cor_func_new = cor_func_raw + cor_func_res

    score_raw = norm(cor_func_raw-ideal_mat)
    score_new = norm(cor_func_new-ideal_mat)
    reward = score_raw - score_new

    if score_new < target_val
        done = true
    else
        done = false
    end

    a1, a2 = action_ 
    state_[a1], state_[a2] = state_[a2], state_[a1]

    return state_, reward, cor_func_new, done
end

function main(iter)
    if debug_test
        cor_list = []
    end

    elapsed_time = @elapsed begin
        ele_list = ele_list_gen(cr_, mn_, co_, ni_, "randchoice")
        cor_func = cor_func_all(ele_list)

        step_count = 0
        cor_func_raw = cor_func
        cor_func_n = cor_func
        while true
            action = [rand(1:192), rand(1:192)]
            ele_list_n, r, cor_func_n, done = swap_step(action, cor_func_raw, ele_list, target_val)
            r_ = exp(r/temperature)
            if rand() <= min(r_, 1) && abs(r) > 0.01
                ele_list = ele_list_n
                cor_func_raw = cor_func_n
            end
            
            if debug_test
                push!(cor_list, norm(cor_func_raw-ideal_mat))
            end

            step_count += 1
            if step_count >= cutoff_iter
                ele_list = ele_list_n
                if benchmark_test == false
                    println("iter: $iter, step: $step_count")
                end
                break
            elseif done
                ele_list = ele_list_n
                if benchmark_test == false
                    println("iter: $iter, step: $step_count")
                end 
                # return ele_list_n, elapsed_time
                break
            end
        end
    end

    if benchmark_test
        return [elapsed_time, step_count]
    elseif debug_test
        println(cor_func_all(ele_list, true))
        return cor_list 
    else
        return ele_list, norm(cor_func_n-ideal_mat)
    end
end