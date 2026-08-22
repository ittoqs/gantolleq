using CSV
using DataFrames
using JSON3
using BSON

const FINAL_LABELS = [
    "sangat positif",
    "positif",
    "netral",
    "negatif",
    "sangat negatif"
]

function normalize_label(raw::AbstractString)
    x = lowercase(strip(raw))
    x = replace(x, r"\s+" => " ")

    # angka 1..5 (opsional; kalau dataset pakai ini)
    if x in ["5","4","3","2","1"]
        return x == "5" ?
               "sangat positif" :
               x == "4" ?
               "positif" :
               x == "3" ?
               "netral" :
               x == "2" ?
               "negatif" :
                          "sangat negatif"
    end

    # sangat positif
    if (occursin("sangat", x) || occursin("strong", x) || occursin("very", x)) &&
       (occursin("posit", x) || x in ["pos","positive"])
        return "sangat positif"
    end
    if x in ["sangat positif","strong positive","very positive","very pos","strong pos","sangat pos"]
        return "sangat positif"
    end

    # positif
    if x in ["positif","positive","pos"]
        return "positif"
    end
    if occursin("posit", x)
        # hindari yang sudah ketangkap kuat sebelumnya
        return "positif"
    end

    # netral
    if x in ["netral","neutral","neu"]
        return "netral"
    end

    # sangat negatif
    if (occursin("sangat", x) || occursin("strong", x) || occursin("very", x)) &&
       (occursin("negat", x) || x in ["neg","negative"])
        return "sangat negatif"
    end
    if x in ["sangat negatif","strong negative","very negative","very neg","strong neg","sangat neg"]
        return "sangat negatif"
    end

    # negatif
    if x in ["negatif","negative","neg"]
        return "negatif"
    end
    if occursin("negat", x) || occursin("neg", x)
        return "negatif"
    end

    return ""
end

function parse_thresholds(s::AbstractString)
    parts = split(s, ",")
    d = Dict{String,Float64}()
    for p in parts
        kv = split(strip(p), "=")
        length(kv) == 2 || continue
        d[strip(kv[1])] = parse(Float64, strip(kv[2]))
    end
    pos_strong = get(d, "pos_strong", 0.7)
    neg_strong = get(d, "neg_strong", 0.7)
    return (pos_strong=pos_strong, neg_strong=neg_strong)
end

function read_texts_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    @assert hasproperty(df, :text) "CSV input harus punya kolom 'text'"
    return String.(df.text)
end

function read_texts_jsonl(path::AbstractString)
    texts = String[]
    open(path, "r", encoding="utf-8") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            obj = JSON3.read(line)
            push!(texts, String(obj["text"]))
        end
    end
    return texts
end

function read_input_texts(path::AbstractString)
    lower = lowercase(path)
    if endswith(lower, ".csv")
        return read_texts_csv(path)
    elseif endswith(lower, ".jsonl") || endswith(lower, ".jsonl.gz")
        # Untuk .jsonl.gz butuh ekstensi gzip;
        # di sini kita asumsikan .jsonl biasa.
        return read_texts_jsonl(path)
    else
        error("Input harus .csv atau .jsonl")
    end
end

function write_outputs(texts::Vector{String}, labels::Vector{String},
                        scores_per_class::Vector{Dict{String,Float64}},
                        out_prefix::AbstractString)
    out_dir = dirname(out_prefix)
    if !isempty(out_dir)
        mkpath(out_dir)
    end

    # CSV
    rows = DataFrame(text=texts, label=labels)

    for cls in FINAL_LABELS
        colname = "score_" * replace(cls, " " => "_")
        rows[!, colname] = [sc[cls] for sc in scores_per_class]
    end

    csv_path = out_prefix * ".csv"
    CSV.write(csv_path, rows)

    # JSON
    json_path = out_prefix * ".json"
    out = [Dict(
        "text" => texts[i],
        "label" => labels[i],
        "scores" => scores_per_class[i]
    ) for i in eachindex(texts)]

    open(json_path, "w", encoding="utf-8") do io
        JSON3.pretty(io, out)
    end

    println("Wrote: $csv_path")
    println("Wrote: $json_path")
end
