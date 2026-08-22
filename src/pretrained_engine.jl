using JSON3

function predict_pretrained(input_path::AbstractString, pretrained_model::AbstractString,
                             thresholds, out_prefix::AbstractString)
    out_tmp = tempname() * ".json"
    py_script = joinpath(@__DIR__, "..", "python", "pretrained_runner.py")

    cmd = `python $(py_script)
            --model $(pretrained_model)
            --input $(input_path)
            --output $(out_tmp)
            --pos_strong $(thresholds.pos_strong)
            --neg_strong $(thresholds.neg_strong)`
    run(cmd)

    try
        # Optimize memory allocation by reading to a byte vector directly
        raw = JSON3.read(read(out_tmp))
        texts = [String(r["text"]) for r in raw]
        labels = [String(r["label"]) for r in raw]

        scores_per_class = Vector{Dict{String,Float64}}()
        for r in raw
            scores = Dict{String,Float64}()
            s = r["scores"]
            for cls in FINAL_LABELS
                scores[cls] = Float64(s[cls])
            end
            push!(scores_per_class, scores)
        end

        write_outputs(texts, labels, scores_per_class, out_prefix)
    finally
        if isfile(out_tmp)
            rm(out_tmp)
        end
    end
end
